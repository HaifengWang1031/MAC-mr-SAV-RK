"""Isolated native-state FEM comparison/replay, or a finer fixed-step temporal reference.

Computation writes raw campaigns under runs/. Reporting is a separate readonly tool.
A source directory is chosen before importing any solver/experiment package.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter, process_time
from typing import Any
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[1]
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(args.source).resolve()))
    import cProfile
    import numpy as np
    from skfem import MeshTri
    from experiments.workflow import provenance, write_json
    from experiments.cylinder_ns.observations import Observations
    from experiments.cylinder_ns.problems import inlet_velocity
    from experiments.periodic_ns.run import make_controller
    from solver.core import State
    from solver.fem import MeshSpace, OpenFlowModel
    from solver.integrate import Result, integrate, step_sizes
    from solver.adaptivity import AdaptiveResult, ScalarErrorControl, integrate_adaptive
    from solver.schemes.registry import make_scheme
    from importlib.metadata import version

    start_cpu, start_wall = process_time(), perf_counter()
    cfg = json.loads((args.input/'config.json').read_text())
    with np.load(args.input/'native.npz', allow_pickle=False) as saved:
        mesh = MeshTri(saved['coordinates'], saved['connectivity'], sort_t=False).with_boundaries(
            {key: saved['boundary_'+key] for key in ('inlet', 'outlet', 'walls', 'cylinder')})
        assert mesh.boundaries is not None
        fixed = np.concatenate([mesh.boundaries[key] for key in ('inlet', 'walls', 'cylinder')])
        options: dict[str, Any] = {'dirichlet_facets': fixed, 'boundary_velocity': lambda x, y: inlet_velocity(x, y, cfg['inlet_max']),
                   'tolerance': cfg['stokes_tolerance'], 'cache_size': cfg['cache_size']}
        if 'stokes_backend' in inspect.signature(OpenFlowModel).parameters:
            options['stokes_backend'] = args.backend
        elif args.backend != 'direct':
            raise ValueError('This source has no recycled backend')
        model = OpenFlowModel(MeshSpace(mesh, cfg['quadrature_order']), cfg['nu'], **options)
        assert np.array_equal(model.component_dofs[0], saved['component_u'])
        assert np.array_equal(model.component_dofs[1], saved['component_v'])
        native = State(float(saved['time']), saved['u'].copy(), saved['v'].copy(), float(saved['r']))
        full = np.zeros(model.velocity_basis.N)
        for values, indices in zip((native.u, native.v), model.component_dofs):
            full[indices] = values
        initial = model.state(native.t, full[model.free_dofs], native.r)
        boundary_difference = max(float(np.max(abs(initial.u-native.u))), float(np.max(abs(initial.v-native.v))))
        assert boundary_difference < 1e-12, boundary_difference
        initial_step = float(saved['initial_step'])
    setup_cpu, setup_wall = process_time()-start_cpu, perf_counter()-start_wall
    observer = Observations(model, cfg['inlet_max'])
    cpu, wall = process_time(), perf_counter()
    observer(initial)
    observation_setup_cpu, observation_setup_wall = process_time()-cpu, perf_counter()-wall
    scheme = make_scheme(cfg['scheme'], cfg['gamma'])
    profile = cProfile.Profile() if args.profile else None
    cpu = process_time()
    if profile is not None:
        profile.enable()
    def observe(state: State) -> None:
        observer(state)
        if (len(observer.rows)-1) % 200 == 0:
            print(f"accepted={len(observer.rows)-1} t={state.t:.12g} r={state.r:.6g}", flush=True)

    result: Result
    requests = (np.linspace(initial.t+max(0., args.duration-2.), initial.t+args.duration, 61).tolist()
                if args.duration >= .1 else [])
    if args.schedule is not None:
        with np.load(args.schedule, allow_pickle=False) as data:
            steps = data['attempt_h'][data['accepted']]
        result = integrate(model, scheme, initial, steps, on_accept=observe)
    elif args.step is not None:
        steps = step_sizes(args.duration, dt=args.step)
        result = integrate(model, scheme, initial, steps, snapshots=requests, on_accept=observe)
    else:
        scalar = None if cfg['scalar_atol'] is None else ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
        result = integrate_adaptive(model, scheme, initial, args.duration, initial_step,
                                    controller=make_controller(cfg['controller'], cfg['scheme']),
                                    scalar_control=scalar, snapshots=requests, on_accept=observe)
    integration_cpu = process_time()-cpu-result.observer_cpu_seconds
    if profile is not None:
        profile.disable()
        profile.dump_stats(str(args.output.with_suffix('.pstats')))
    if isinstance(result, AdaptiveResult):
        attempt_t = [float(a['t']) for a in result.attempts]
        attempt_h = [float(a['step']) for a in result.attempts]
        accepted = [bool(a['accepted']) for a in result.attempts]
    else:
        attempt_t, attempt_h, accepted = result.times[:-1], steps.tolist(), [True]*len(steps)
    np.savez(args.output.with_suffix('.npz'), u=result.final.u, v=result.final.v, r=result.final.r,
             times=result.times, attempt_t=attempt_t, attempt_h=attempt_h, accepted=accepted,
             observation_time=[row['time'] for row in observer.rows],
             drag=[row['drag'] for row in observer.rows], lift=[row['lift'] for row in observer.rows],
             pressure=[row['pressure_difference'] for row in observer.rows],
             scalar_history=[d['r'] for d in result.diagnostics],
             physical_pressure=observer.pressure if observer.pressure is not None else np.empty(0),
             snapshot_times=result.snapshot_times,
             snapshot_u=np.stack([s.u for s in result.snapshots]) if result.snapshots else np.empty((0, *result.final.u.shape)),
             snapshot_v=np.stack([s.v for s in result.snapshots]) if result.snapshots else np.empty((0, *result.final.v.shape)))
    metrics = model.performance_metrics() if hasattr(model, 'performance_metrics') else {}
    metrics.update(setup_cpu_seconds=setup_cpu, setup_wall_seconds=setup_wall,
                   integration_cpu_seconds=integration_cpu, integration_wall_seconds=result.seconds,
                   observer_cpu_seconds=observation_setup_cpu+result.observer_cpu_seconds,
                   observer_wall_seconds=observation_setup_wall+result.observer_wall_seconds,
                   accepted_steps=len(result.times)-1, rejected_steps=int(np.sum(np.logical_not(accepted))),
                   max_stage_residual=max((s['residual'] for step in result.stages for s in step), default=0.),
                   max_continuity=max(d['continuity_residual'] for d in result.diagnostics),
                   max_abs_r=max(abs(d['r']) for d in result.diagnostics),
                   max_physical_divergence=max(d['divergence_inf'] for d in result.diagnostics))
    source = provenance(); source['packages']['scikit-fem'] = version('scikit-fem')
    write_json(args.output.with_suffix('.json'), {
        'status': result.status, 'error': result.error, 'source': source, 'backend': args.backend,
        'duration': args.duration, 'step': args.step, 'schedule': None if args.schedule is None else str(args.schedule),
        'initial_time': initial.t, 'final_time': result.final.t, 'boundary_roundoff_adjustment': boundary_difference,
        'profiled': args.profile, 'metrics': metrics,
        'total_body_cpu_seconds': process_time()-start_cpu, 'total_body_wall_seconds': perf_counter()-start_wall,
        'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)})
    if result.status != 'complete':
        raise RuntimeError(result.error)


def _campaign(args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(PROJECT))
    import h5py
    import numpy as np
    from experiments.cylinder_ns.run import load_cylinder
    from experiments.cylinder_ns.config import DEFAULTS, effective_config
    from experiments.cylinder_ns.representation import validate
    from experiments.cylinder_ns.benchmark import cycle_statistics
    from experiments.workflow import digest, write_json

    if args.duration <= 0. or not args.duration < float('inf') or args.repeats < 1:
        raise ValueError('Positive finite duration and repeats required')
    if args.command == 'reference' and not args.adaptive and (args.step is None or not 0. < args.step < float('inf')):
        raise ValueError('Reference requires a positive finite fixed step or adaptive control')
    if args.control_config is not None and not args.adaptive:
        raise ValueError('A control config requires an adaptive reference')
    cfg, original = load_cylinder(args.run, require_complete=True)
    seed_cfg = cfg.copy()
    if cfg['forcing'] != 'zero':
        raise ValueError('Native warm-start experiment requires the autonomous cylinder problem')
    if args.adaptive:
        controls = json.loads(args.control_config.read_text()) if args.control_config is not None else {}
        if not isinstance(controls, dict) or set(controls)-{'controller', 'scalar_atol', 'initial_step'}:
            raise ValueError('Reference control config accepts controller/scalar_atol/initial_step only')
        base = {key: cfg[key] for key in DEFAULTS if key not in ('dt', 'snapshots')}
        base.update(mode='adaptive', T=args.duration, snapshots=[], initial_step=cfg.get('initial_step') or .001)
        base.update(controls)
        cfg = effective_config(base)
    if args.command == 'reference':
        cfg['stokes_backend'] = args.backend
    refined = args.resolution is not None or args.circle_segments is not None
    family = ('cylinder_refinement_reference' if refined else 'cylinder_temporal_reference') \
        if args.command == 'reference' else 'fem_adaptive_benchmark'
    directory = args.root/'runs'/family/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    inputs = directory/'input'; inputs.mkdir()
    with h5py.File(args.run/'results.h5', 'r') as data:
        vb, _ = validate(data, seed_cfg, original)
        accepted_steps = data['attempts/step'][:][data['attempts/accepted'][:]] if 'attempts' in data else data['dt'][:]
        seed_cycles = cycle_statistics(data['observations/time'][:], data['observations/drag'][:],
                                       data['observations/lift'][:], data['observations/pressure_difference'][:],
                                       mean_velocity=2*cfg['inlet_max']/3, diameter=2*cfg['radius'])
        typical_step = cfg['initial_step'] if args.adaptive else float(np.median(accepted_steps[-100:]))
        mesh = vb.mesh
        parts = vb.split_indices()
        u, v = data['final/u'][:], data['final/v'][:]
        if refined:
            from scipy.spatial import ConvexHull
            from solver.fem import CylinderSpace, OpenFlowModel
            from experiments.cylinder_ns.problems import inlet_velocity
            space = CylinderSpace(args.resolution if args.resolution is not None else cfg['resolution'],
                                  args.circle_segments, cfg['quadrature_order'])
            assert mesh.boundaries is not None and space.mesh.boundaries is not None
            fixed = np.concatenate([space.mesh.boundaries[key] for key in ('inlet', 'walls', 'cylinder')])
            target = OpenFlowModel(space, cfg['nu'], dirichlet_facets=fixed,
                                    boundary_velocity=lambda x,y: inlet_velocity(x,y,cfg['inlet_max']))
            old_full = np.zeros(vb.N)
            for values, indices in zip((u, v), parts): old_full[indices] = values
            query = target.velocity_basis.doflocs[:, target.component_dofs[0]]
            body_nodes = np.unique(mesh.facets[:, mesh.boundaries['cylinder']])
            hull = ConvexHull(mesh.p[:, body_nodes].T)
            outside = np.any(hull.equations[:, :2]@query+hull.equations[:, 2, None] > 1e-11, axis=0)
            sampled = np.zeros((2, query.shape[1]))
            sampled[:, outside] = vb.interpolator(old_full)(query[:, outside])
            full = np.zeros(target.velocity_basis.N)
            for values, indices in zip(sampled, target.component_dofs): full[indices] = values
            projected = target.solve(full[target.free_dofs], mass=1., viscosity=0.)
            mapped = target.state(float(data['final/t'][()]), projected.velocity, float(data['final/r'][()]))
            assert target.max_abs(target.apply_D(projected.velocity)) < 1e-9
            mesh, parts, u, v = space.mesh, target.component_dofs, mapped.u, mapped.v
            cfg.update(resolution=space.resolution, circle_segments=space.circle_segments,
                       initial_condition='named_native_state_interpolated_zero_extended_then_affine_weak_Leray')
        assert mesh.boundaries is not None
        np.savez(inputs/'native.npz', coordinates=mesh.p, connectivity=mesh.t,
                 **{'boundary_'+key: mesh.boundaries[key] for key in ('inlet','outlet','walls','cylinder')},
                 component_u=parts[0], component_v=parts[1], u=u, v=v,
                 r=data['final/r'][()], time=data['final/t'][()], initial_step=typical_step)
    cfg['initial_condition'] = ('named_native_state_interpolated_zero_extended_then_affine_weak_Leray' if refined
                                else 'named_native_state')
    write_json(inputs/'config.json', cfg)
    manifest: dict = {'experiment': family, 'run_id': directory.name, 'status': 'running',
                      'seed_run': str(args.run.resolve()), 'seed_run_id': args.run.name,
                      'seed_manifest_sha256': digest(args.run/'manifest.json'),
                      'seed_results_sha256': original['results_sha256'], 'duration': args.duration,
                      'fixed_step': args.step, 'repeats': args.repeats, 'seed_cycles': seed_cycles,
                      'initial_step': typical_step,
                      'mesh_refinement': {'enabled': refined, 'resolution': cfg['resolution'],
                                          'circle_segments': cfg['circle_segments']},
                      'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'input_sha256': {p.name: digest(p) for p in inputs.iterdir()}, 'jobs': []}
    write_json(directory/'manifest.json', manifest)
    if args.prepare_only:
        manifest['status'] = 'prepared'
        write_json(directory/'manifest.json', manifest)
        return directory
    env = os.environ.copy(); env.update({key:'1' for key in THREADS}); env['MPLBACKEND']='Agg'
    sources = [('baseline', args.baseline, 'direct'), ('direct', args.source, 'direct'),
               ('recycled', args.source, 'recycled')] if args.command == 'compare' else [(args.backend, args.source, args.backend)]

    def launch(label: str, source: Path, backend: str, mode: str, repeat: int,
               schedule: Path | None = None, profile: bool = False) -> None:
        prefix = directory/f'{mode}-{label}-{repeat}'
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--source', str(source.resolve()),
                   '--input', str(inputs.resolve()), '--output', str(prefix.resolve()), '--backend', backend,
                   '--duration', str(args.duration)]
        if schedule is not None: command += ['--schedule', str(schedule.resolve())]
        if args.step is not None: command += ['--step', str(args.step)]
        if profile: command += ['--profile']
        start = perf_counter()
        with prefix.with_suffix('.log').open('w') as stream:
            process = subprocess.run(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
        if process.returncode:
            raise RuntimeError(f'Benchmark worker failed: {prefix.with_suffix(".log")}')
        suffixes = ('.json', '.npz', '.log', '.pstats') if profile else ('.json', '.npz', '.log')
        job = {'label':label, 'mode':mode, 'repeat':repeat, 'profiled':profile, 'command':command,
               'process_wall_seconds':perf_counter()-start,
               'artifacts': {suffix: digest(prefix.with_suffix(suffix)) for suffix in suffixes}}
        manifest['jobs'].append(job)
        write_json(directory/'manifest.json', manifest)
        print(f'{mode} {label} {repeat}: {job["process_wall_seconds"]:.3f}s', flush=True)
    try:
        for repeat in range(args.repeats):
            order = sources[repeat % len(sources):]+sources[:repeat % len(sources)]
            for label, source, backend in order:
                launch(label, source, backend, 'fixed' if args.step is not None else 'adaptive', repeat)
        if args.command == 'compare':
            schedule = directory/'adaptive-direct-0.npz'
            for repeat in range(args.repeats):
                order = sources[repeat % len(sources):]+sources[:repeat % len(sources)]
                for label, source, backend in order:
                    launch(label, source, backend, 'replay', repeat, schedule=schedule)
            if args.repeats >= 3:
                for label, source, backend in sources:
                    launch(label, source, backend, 'profile', 0, profile=True)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(directory/'manifest.json', manifest)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    worker = commands.add_parser('worker')
    for key in ('source','input','output'):
        worker.add_argument('--'+key, type=Path, required=True)
    worker.add_argument('--backend', choices=('direct','recycled'), required=True)
    worker.add_argument('--duration', type=float, required=True)
    worker.add_argument('--step', type=float)
    worker.add_argument('--schedule', type=Path)
    worker.add_argument('--profile', action='store_true')
    for name in ('compare','reference'):
        command = commands.add_parser(name)
        command.add_argument('--run', type=Path, required=True)
        command.add_argument('--source', type=Path, default=PROJECT)
        command.add_argument('--root', type=Path, default=PROJECT)
        command.add_argument('--duration', type=float, default=.05 if name=='compare' else 3.)
        command.add_argument('--repeats', type=int, default=3 if name=='compare' else 1)
        command.add_argument('--timeout', type=float, default=7200.)
        if name == 'compare':
            command.add_argument('--baseline', type=Path, required=True)
            command.set_defaults(step=None, backend='direct', resolution=None, circle_segments=None,
                                 prepare_only=False, adaptive=False, control_config=None)
        else:
            method = command.add_mutually_exclusive_group(required=True)
            method.add_argument('--step', type=float)
            method.add_argument('--adaptive', action='store_true')
            command.add_argument('--control-config', type=Path)
            command.add_argument('--backend', choices=('direct','recycled'), default='direct')
            command.add_argument('--prepare-only', action='store_true')
            command.add_argument('--resolution', type=float)
            command.add_argument('--circle-segments', type=int)
            command.set_defaults(baseline=None)
    args = parser.parse_args()
    if args.command == 'worker': _worker(args)
    else: print(_campaign(args))


if __name__ == '__main__': main()

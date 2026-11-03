"""Run and verify DFG2D-2 records; analysis is a separate readonly entry point."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import h5py

from experiments.fem_ns.attempts import save_attempts
from experiments.controllers import make_controller
from experiments.records import PROJECT, begin_run, finish_run, provenance, reusable_run, save_result, fail_run, digest
from experiments.fem_backends import solver_provenance
from solver.adaptivity import AdaptiveResult, ScalarErrorControl, integrate_adaptive
from solver.core import State
from solver.integrate import Result, integrate
from solver.schemes.registry import make_scheme, scheme_info
from .config import effective_config
from .records import load_cylinder
from .observations import Observations
from .problems import build_model
from .representation import save


def run(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    total_wall, total_cpu = perf_counter(), process_time()
    cfg = effective_config(config)
    source = provenance()
    solver_provenance(source, cfg)
    source['packages']['scikit-fem'] = version('scikit-fem')
    source['packages']['pymetis'] = version('pymetis')
    if cfg.get('stokes_factor_backend') == 'pardiso':
        source['packages'].update({name: version(name) for name in ('pypardiso', 'mkl')})
    parent = Path(root)/'runs'/'cylinder_ns'
    reused = reusable_run(parent, cfg, source, reader=lambda path: load_cylinder(path, require_complete=True), rerun=rerun)
    if reused is not None:
        return reused
    directory, manifest = begin_run(parent, cfg, source, metadata={'experiment': 'cylinder_ns', 'rerun': rerun},
                                    started_key='started')
    (directory/'run.log').write_text(f'Started {manifest["started"]}\n')
    if 'min_step_policy' in cfg:
        with (directory/'run.log').open('a') as stream:
            stream.write(f'min_step_policy={cfg["min_step_policy"]}; finite over-tolerance floor steps remain recorded\n')
    latest_r = cfg['r0']

    def progress(count: int, t: float, elapsed: float) -> None:
        if count % cfg['log_every'] == 0 or abs(t-cfg['T']) < 1e-12:
            with (directory/'run.log').open('a') as stream:
                last = observations.rows[-1]
                step = last['time']-observations.rows[-2]['time']
                stream.write(f'accepted={count} t={t:.17g} integration_wall={elapsed:.6f}s '
                             f'dt={step:.9g} r={latest_r:.9g} '
                             f'CD={last["drag"]:.7f} CL={last["lift"]:.7f} '
                             f'dp={last["pressure_difference"]:.7f}\n')

    try:
        cpu, wall = process_time(), perf_counter()
        model, initial = build_model(cfg)
        setup_cpu, setup_wall = process_time()-cpu, perf_counter()-wall
        observations = Observations(model, cfg['inlet_max'])
        cpu, wall = process_time(), perf_counter()
        observations(initial)
        def observe(state: State) -> None:
            nonlocal latest_r
            observations(state)
            latest_r = state.r
        observation_setup_cpu, observation_setup_wall = process_time()-cpu, perf_counter()-wall
        selected = make_scheme(cfg['scheme'], cfg['gamma'])
        cpu = process_time()
        result: Result
        if cfg['mode'] == 'adaptive':
            scalar = None if cfg['scalar_atol'] is None else ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
            result = integrate_adaptive(model, selected, initial, cfg['T'], cfg['initial_step'],
                                        controller=make_controller(cfg['controller'], cfg['scheme']),
                                        scalar_control=scalar, snapshots=cfg['snapshots'], progress=progress,
                                        on_accept=observe, min_step_policy=cfg.get('min_step_policy', 'fail'))
        else:
            result = integrate(model, selected, initial, cfg['actual_steps'], snapshots=cfg['snapshots'],
                               progress=progress, on_accept=observe)
        metrics = {**model.performance_metrics(),
                   'setup_cpu_seconds': setup_cpu, 'setup_wall_seconds': setup_wall,
                   'integration_cpu_seconds': process_time()-cpu-result.observer_cpu_seconds,
                   'integration_wall_seconds': result.seconds,
                   'observer_cpu_seconds': observation_setup_cpu+result.observer_cpu_seconds,
                   'observer_wall_seconds': observation_setup_wall+result.observer_wall_seconds,
                   'accepted_steps': len(result.times)-1,
                   'rejected_steps': sum(not a['accepted'] for a in result.attempts) if isinstance(result, AdaptiveResult) else 0,
                   'forced_min_step_acceptances': sum(bool(a['accepted']) and float(a['error']) > 1.
                        for a in result.attempts) if isinstance(result, AdaptiveResult) else 0,
                   'velocity_dofs': int(len(model.free_dofs)), 'pressure_dofs': int(model.pressure_basis.N),
                   'triangles': int(model.velocity_basis.mesh.t.shape[1]),
                   'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform == 'darwin' else 1024),
                   'total_compute_wall_seconds': perf_counter()-total_wall,
                   'total_compute_cpu_seconds': process_time()-total_cpu}
        finalization = perf_counter()
        pending = directory/'results.pending.h5'
        info = scheme_info(cfg['scheme'])
        save_result(pending, result, metrics, n_stages=info.n_stages, n_roots=info.n_roots,
                    pressure_semantics=cfg['pressure_semantics'])
        with h5py.File(pending, 'a') as data:
            save(data, model, cfg, observations, result)
            if isinstance(result, AdaptiveResult):
                save_attempts(data, result.attempts)
        pending.replace(directory/'results.h5')
        checksum = digest(directory/'results.h5')
        metrics['finalization_wall_seconds'] = perf_counter()-finalization
        metrics['workflow_wall_seconds'] = perf_counter()-total_wall
        finish_run(directory, manifest, status=result.status, error=result.error, metrics=metrics, results_sha256=checksum)
        with (directory/'run.log').open('a') as stream:
            stream.write(f'{result.status}: {result.error}\n{json.dumps(metrics, indent=2)}\n')
    except Exception as error:
        fail_run(directory, manifest, error)
        with (directory/'run.log').open('a') as stream:
            stream.write(manifest['error']+'\n')
        raise
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    directory = run(json.loads(args.config.read_text()), root=args.root, rerun=args.rerun)
    print(directory)
    _, manifest = load_cylinder(directory)
    if manifest['status'] != 'complete':
        raise SystemExit(f'Cylinder computation failed: {manifest["error"]}')


if __name__ == '__main__':
    main()

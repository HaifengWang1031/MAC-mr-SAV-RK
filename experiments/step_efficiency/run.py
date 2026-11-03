"""Compare warm steps of all serial backends and time schemes at the same N.

Controls: domain, viscosity, nominal step, threads, scheme and repeat count.
Observables: CPU/wall cost, factor/iteration counts, full stage checks and fields.
The fixed schedule excludes its cached factor; new variable shifts are charged.
Different boundary conditions and N definitions prevent an equal-accuracy claim.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, cast
from uuid import uuid4

import numpy as np

from solver.core import State
from solver.model import Model
from solver.mac.kernels import warmup as warmup_mac
from solver.channel.kernels import warmup as warmup_channel
from solver.fourier import FourierModel, FourierSpace
from solver.spectral.assembly import Space
from solver.spectral.model import SpectralModel
from solver.schemes.registry import SCHEMES, make_scheme, scheme_info
from experiments.records import PROJECT, digest, provenance, write_json
from experiments.legendre_mac_benchmark.run import build_model as build_mac, schedules
from experiments.legendre_optimization.run import build_model as build_legendre
from experiments.channel_ns.problems import build_model as build_channel
from experiments.periodic_ns.problems import Manufactured as PeriodicManufactured

CONFIG = Path(__file__).parent/'configs/n128.json'
BACKENDS = ('mac_direct', 'mac_tensor', 'legendre_direct', 'legendre_tensor',
            'channel_direct', 'channel_fft', 'fourier_leray')
COUNTERS = ('factorizations', 'factorization_seconds', 'solve_seconds', 'check_seconds',
            'right_hand_sides', 'rhs_count', 'solve_calls', 'solves', 'total_iterations',
            'schur_rhs', 'helmholtz_applications')


def effective_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = json.loads(CONFIG.read_text())
    if set(config)-set(defaults):
        raise ValueError('Unknown step efficiency configuration')
    cfg = {**defaults, **config}
    for key in ('N', 'steps', 'warmup_steps', 'repeats', 'cache_size', 'max_iterations'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'Invalid {key}')
    for key, allowed in (('backends', BACKENDS), ('schemes', SCHEMES)):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]) or set(cfg[key])-set(allowed):
            raise ValueError(f'Invalid {key}')
    for key in ('lx', 'ly', 'nu', 'amplitude', 'fixed_step', 'stokes_tolerance', 'krylov_tolerance'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not np.isfinite(cfg['gamma']) or cfg['gamma'] < 0:
        raise ValueError('Invalid gamma')
    if type(cfg['random_seed']) is not int or cfg['random_seed'] < 0:
        raise ValueError('Invalid random seed')
    low, high = cfg['random_multiplier_interval']
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError('Invalid variable-step multipliers')
    if cfg['N'] < 3 or any(type(cfg[k]) is not int or cfg[k] < 0 for k in ('dealias', 'projection_extra')):
        raise ValueError('Invalid resolution/quadrature')
    cfg['actual_steps'] = {key: value.tolist() for key, value in schedules(cfg).items()}
    return cfg


def build_model(config: dict[str, Any], backend: str) -> tuple[Model, State]:
    """Compatible smooth MMS; forces assembled once in each natural load space."""
    n = config['N']
    if backend.startswith('mac_'):
        return build_mac(config, backend, n)
    if backend.startswith('legendre_'):
        return build_legendre({**config, 'optimized_backend': backend.removeprefix('legendre_')},
                              n, (Space, SpectralModel))
    if backend.startswith('channel_'):
        model, initial, _ = build_channel({**config, 'nx': n, 'ny': n, 'case': 'mms',
                                          'backend': backend.removeprefix('channel_'), 'r0': 0.})
        return model, initial
    if backend != 'fourier_leray':
        raise ValueError(f'Unknown backend {backend}')
    space = FourierSpace(n, n, config['lx'], config['ly'])
    reference = PeriodicManufactured(space, config['nu'], config['amplitude'])
    base, stiffness, convection, gradient = [space.to_spectral(value) for value in
        (reference.base, reference.negative_laplacian, reference.convection, reference.pressure_gradient)]

    def force(t: float) -> Any:
        a = reference.amplitude(t)
        return reference.derivative(t)*base+config['nu']*a*stiffness+a*a*convection+a*gradient

    periodic = FourierModel(space, config['nu'], force=force, tolerance=config['stokes_tolerance'])
    velocity = reference.exact(0.)
    return periodic, periodic.state(0., space.velocity(velocity[0], velocity[1]))


def backend_counters(model: Model) -> dict[str, float]:
    backend = getattr(model, 'stokes', getattr(model, 'backend', None))
    return {key: float(getattr(backend, key)) for key in COUNTERS if hasattr(backend, key)}


def run_case(config: dict[str, Any], backend: str, name: str, schedule: str
             ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cpu, wall = process_time(), perf_counter()
    model, initial = build_model(config, backend)
    setup_cpu, setup_wall = process_time()-cpu, perf_counter()-wall
    scheme = make_scheme(name, config['gamma'])
    try:
        # Compile/load kernels and exercise all scheme paths, then restart at t=0.
        cpu, wall = process_time(), perf_counter()
        state = initial
        for _ in range(config['warmup_steps']):
            state = scheme.step(model, state, config['fixed_step']).state
            model.diagnostics(state)
        warmup_cpu, warmup_wall = process_time()-cpu, perf_counter()-wall
        before = backend_counters(model)
        state = initial
        timing, diagnostics, checks, times = [], [], [], [state.t]
        diagnostic_keys: list[str] = []
        last = None
        for dt in config['actual_steps'][schedule]:
            cpu, wall = process_time(), perf_counter()
            last = scheme.step(model, state, dt)
            step_cpu, step_wall = process_time()-cpu, perf_counter()-wall
            state = last.state
            cpu, wall = process_time(), perf_counter()
            diagnostic = model.diagnostics(state)
            diagnostic_cpu, diagnostic_wall = process_time()-cpu, perf_counter()-wall
            diagnostic_keys = sorted(diagnostic)
            diagnostics.append([diagnostic[key] for key in diagnostic_keys])
            timing.append([step_cpu, step_wall, diagnostic_cpu, diagnostic_wall])
            checks.append([[stage.residual, stage.continuity_residual, stage.divergence_inf,
                            stage.scalar_residual] for stage in last.stages])
            times.append(state.t)
        assert last is not None
        arrays = {
            'u': state.u, 'v': state.v, 'pressure': np.array([s.pressure for s in last.stages]),
            'stage_r': np.array([s.r for s in last.stages]), 'timing': np.array(timing),
            'checks': np.array(checks), 'diagnostics': np.array(diagnostics), 'times': np.array(times),
            'r': np.array(state.r),
        }
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError('Nonfinite benchmark result')
        after = backend_counters(model)
        return {'setup_cpu_seconds': setup_cpu, 'setup_wall_seconds': setup_wall,
                'warmup_cpu_seconds': warmup_cpu, 'warmup_wall_seconds': warmup_wall,
                'counters_before': before, 'counters': {key: after[key]-value for key, value in before.items()},
                'diagnostic_keys': diagnostic_keys, 'n_stages': scheme_info(name).n_stages,
                'velocity_dofs': int(np.asarray(model.vector(state)).size), 'final_time': state.t,
                'final_r': state.r}, arrays
    finally:
        if hasattr(model, 'close'):
            cast(Any, model).close()


def run_benchmark(config: dict[str, Any], *, root: Path = PROJECT) -> Path:
    cfg = effective_config(config)
    directory = root/'runs'/'step_efficiency'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    write_json(directory/'config.json', cfg)
    manifest: dict[str, Any] = {'experiment': 'step_efficiency', 'schema_version': 1,
        'status': 'running', 'source': provenance(), 'config_sha256': digest(directory/'config.json')}
    write_json(directory/'manifest.json', manifest)
    rows: list[dict[str, Any]] = []
    (directory/'run.log').write_text(f'Started {directory.name}\n')
    print(f'Run: {directory}', flush=True)
    try:
        # The global Numba entry points are exercised outside every measured case.
        warmup_mac()
        warmup_channel()
        for repeat in range(cfg['repeats']):
            backends = cfg['backends'] if repeat % 2 == 0 else list(reversed(cfg['backends']))
            for backend in backends:
                for name in cfg['schemes']:
                    for schedule in ('fixed', 'variable'):
                        metrics, fields = run_case(cfg, backend, name, schedule)
                        filename = f'{backend}-{name}-{schedule}-{repeat}.npz'
                        temporary = directory/(filename+'.tmp')
                        with temporary.open('wb') as stream:
                            np.savez_compressed(stream, **cast(Any, fields))
                        temporary.replace(directory/filename)
                        rows.append({'backend': backend, 'scheme': name, 'schedule': schedule,
                                     'repeat': repeat, 'fields': filename,
                                     'fields_sha256': digest(directory/filename), **metrics})
                        write_json(directory/'results.json', {'rows': rows})
                        gc.collect()
                message = f'repeat={repeat+1}/{cfg["repeats"]} backend={backend} completed={len(rows)}'
                print(message, flush=True)
                with (directory/'run.log').open('a') as log:
                    log.write(message+'\n')
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(directory/'results.json', {'rows': rows})
        manifest['results_sha256'] = digest(directory/'results.json')
        manifest['completed_cases'] = len(rows)
        write_json(directory/'manifest.json', manifest)
        with (directory/'run.log').open('a') as log:
            log.write(f'{manifest["status"]}: {manifest.get("error", "")}\n')
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(run_benchmark(json.loads(args.config.read_text()), root=args.root))


if __name__ == '__main__':
    main()

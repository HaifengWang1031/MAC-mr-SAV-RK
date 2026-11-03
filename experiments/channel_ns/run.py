"""Compute channel runs; centralized configuration and atomic result finalization."""
import argparse
import json
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import h5py
import numpy as np

from solver.adaptivity import AdaptiveResult, ScalarErrorControl, integrate_adaptive
from solver.channel import ChannelGrid
from solver.channel.kernels import warmup
from solver.integrate import Result, integrate, step_sizes
from solver.mac.stokes import DirectStokes
from solver.channel import FFTStokes
from solver.schemes.registry import scheme_info
from experiments.controllers import controller_settings, make_controller
from experiments.records import PROJECT, begin_run, finish_run, provenance, reusable_run, save_result
from .records import load_channel
from .problems import CASES, SCHEMES, build_model, scheme

DEFAULTS: dict[str, Any] = {
    'case': 'perturbed', 'scheme': 'sdirk3_mrsav', 'nx': 32, 'ny': 24,
    'lx': 2*np.pi, 'ly': 1., 'nu': .1, 'driving': .1, 'epsilon': .01,
    'amplitude': .1, 'gamma': 1., 'r0': 0., 'T': .1, 'dt': .01,
    'mode': 'fixed', 'initial_step': .01, 'controller': None, 'scalar_atol': None,
    'snapshots': None, 'log_every': 100, 'cache_size': 4, 'stokes_tolerance': 1e-10,
    'backend': 'direct', **ChannelGrid(3, 3).metadata(),
}


def effective_config(config: dict) -> dict:
    if set(config)-set(DEFAULTS)-{'steps'}: raise ValueError('Unknown channel configuration key')
    cfg = {**DEFAULTS, **config}
    grid = ChannelGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    for key, value in grid.metadata().items():
        if cfg[key] != value: raise ValueError(f'Channel requires {key}={value}')
    if cfg['backend'] not in ('direct', 'fft'): raise ValueError('Unknown channel backend')
    if cfg['case'] not in CASES or cfg['scheme'] not in SCHEMES:
        raise ValueError('Unknown channel case or scheme')
    for key in ('log_every', 'cache_size'):
        if type(cfg[key]) is not int or cfg[key] < 1: raise ValueError(f'Invalid {key}')
    numeric = ('nu', 'T', 'driving', 'epsilon', 'amplitude', 'gamma', 'r0', 'stokes_tolerance')
    if not np.isfinite([cfg[key] for key in numeric]).all() \
            or min(cfg['nu'], cfg['T'], cfg['amplitude'], cfg['stokes_tolerance']) <= 0 \
            or min(cfg['gamma'], cfg['epsilon']) < 0:
        raise ValueError('Invalid channel physical or solver parameters')
    if cfg['case'] in ('mms', 'discrete_mms') and cfg['r0'] != 0:
        raise ValueError('MMS requires r0=0')
    if cfg['case'] == 'ts_wave' and cfg['driving'] <= 0:
        raise ValueError('TS wave requires positive streamwise driving')
    if cfg['mode'] == 'fixed':
        if cfg['controller'] is not None or cfg['scalar_atol'] is not None:
            raise ValueError('Controller options require adaptive mode')
        if 'steps' in config:
            if 'dt' in config: raise ValueError('Use dt or steps, not both')
            cfg['dt'] = None
        cfg['actual_steps'] = step_sizes(cfg['T'], dt=cfg['dt'], steps=cfg.get('steps')).tolist()
    elif cfg['mode'] == 'adaptive':
        if 'steps' in config or 'dt' in config:
            raise ValueError('Adaptive mode uses initial_step instead of dt/steps')
        options = {'kind': 'PI', 'atol': 1e-8, 'rtol': 5e-5, **(cfg['controller'] or {})}
        control = make_controller(options, cfg['scheme'])
        if not np.isfinite(cfg['initial_step']) or cfg['initial_step'] < control.min_step:
            raise ValueError('Invalid initial_step')
        cfg['controller'] = controller_settings(control)
        cfg['dt'], cfg['actual_steps'] = None, None
        if cfg['scalar_atol'] is not None:
            if cfg['scheme'] != 'sdirk3_mrsav': raise ValueError('Scalar control requires sdirk3_mrsav')
            ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
    else:
        raise ValueError('Unknown integration mode')
    requests = [0., cfg['T']] if cfg['snapshots'] is None else list(cfg['snapshots'])
    if any(not np.isfinite(t) or t < 0 or t > cfg['T'] for t in requests):
        raise ValueError('Snapshot outside channel integration interval')
    cfg.update(snapshots=requests, pressure_semantics=scheme_info(cfg['scheme']).pressure_semantics,
               forcing=('independent_numpy_discrete_mms' if cfg['case'] == 'discrete_mms' else
                        'analytic_continuous_mms' if cfg['case'] == 'mms' else 'uniform_streamwise_acceleration'),
               initial_condition=('discrete_vertex_curl' if cfg['case'] in ('mms', 'discrete_mms') else
                                  'zero' if cfg['case'] == 'startup' else
                                  'discrete_poiseuille_plus_TS_eigenmode' if cfg['case'] == 'ts_wave' else
                                  'discrete_poiseuille_plus_vertex_curl' if cfg['case'] == 'perturbed' else
                                  'discrete_poiseuille_equilibrium'))
    return cfg


def run(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    cfg = effective_config(config)
    source = provenance()
    parent = Path(root)/'runs'/'channel_ns'
    reused = reusable_run(parent, cfg, source, reader=lambda path: load_channel(path, require_complete=True), rerun=rerun)
    if reused is not None:
        return reused
    directory, manifest = begin_run(parent, cfg, source, metadata={'experiment': 'channel_ns'},
                                    started_key='started')
    (directory/'run.log').write_text(f'Started {manifest["started"]}\n{json.dumps(cfg, indent=2)}\n')
    def progress(count: int, t: float, elapsed: float) -> None:
        if count % cfg['log_every'] == 0 or abs(t-cfg['T']) < 1e-12:
            with (directory/'run.log').open('a') as stream:
                stream.write(f'accepted={count} t={t:.17g} wall={elapsed:.3f}s\n')
    cpu = process_time()
    warmup_wall = warmup()
    warmup_cpu = process_time()-cpu
    cpu, wall = process_time(), perf_counter()
    model, initial, _ = build_model(cfg)
    setup_cpu, setup_wall = process_time()-cpu, perf_counter()-wall
    selected = scheme(cfg['scheme'], cfg['gamma'])
    cpu = process_time()
    result: Result
    if cfg['mode'] == 'adaptive':
        scalar = None if cfg['scalar_atol'] is None else ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
        result = integrate_adaptive(model, selected, initial, cfg['T'], cfg['initial_step'],
                                    controller=make_controller(cfg['controller'], cfg['scheme']),
                                    scalar_control=scalar, snapshots=cfg['snapshots'], progress=progress)
    else:
        result = integrate(model, selected, initial, cfg['actual_steps'], snapshots=cfg['snapshots'], progress=progress)
    assert isinstance(model.backend, (DirectStokes, FFTStokes))
    metrics = {'warmup_cpu_seconds': warmup_cpu, 'warmup_wall_seconds': warmup_wall,
               'setup_cpu_seconds': setup_cpu, 'setup_wall_seconds': setup_wall,
               'integration_cpu_seconds': process_time()-cpu, 'integration_wall_seconds': result.seconds,
               'factorizations': model.backend.factorizations,
               'factorization_wall_seconds': model.backend.factorization_seconds,
               'linear_solve_wall_seconds': model.backend.solve_seconds,
               'accepted_steps': len(result.times)-1,
               'rejected_steps': sum(not a['accepted'] for a in result.attempts) if isinstance(result, AdaptiveResult) else 0}
    pending = directory/'results.pending.h5'
    info = scheme_info(cfg['scheme'])
    save_result(pending, result, metrics, n_stages=info.n_stages, n_roots=info.n_roots)
    with h5py.File(pending, 'a') as data:
        for key, value in model.grid.metadata().items(): data.attrs[key] = value
        data.attrs['pressure_semantics'] = cfg['pressure_semantics']
        if cfg['case'] == 'ts_wave':
            from .ts_mode import discrete_mode
            seed = discrete_mode(model.grid, cfg['nu'], cfg['driving'])
            group = data.create_group('initial_mode')
            for label, setting in seed.metadata().items(): group.attrs[label] = setting
            group.create_dataset('unit_velocity', data=seed.velocity)
        if isinstance(result, AdaptiveResult) and result.attempts:
            for key in result.attempts[0]:
                values = [a[key] for a in result.attempts]
                dtype = h5py.string_dtype('utf-8') if isinstance(values[0], str) else None
                data.create_dataset('attempts/'+key, data=values, dtype=dtype)
    pending.replace(directory/'results.h5')
    finish_run(directory, manifest, status=result.status, error=result.error, metrics=metrics)
    with (directory/'run.log').open('a') as stream:
        stream.write(f'{result.status}: {result.error}\n{json.dumps(metrics, indent=2)}\n')
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    directory = run(json.loads(args.config.read_text()), root=args.root, rerun=args.rerun)
    print(directory)
    _, manifest = load_channel(directory)
    if manifest['status'] != 'complete': raise SystemExit(f"Run failed: {manifest['error']}")


if __name__ == '__main__': main()

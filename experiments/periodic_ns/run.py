"""Run periodic NS; preserve the existing no-slip workflow and result schema."""
import argparse
import json
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import h5py
import numpy as np

from solver.adaptivity import AdaptiveResult, ScalarErrorControl, integrate_adaptive
from solver.fourier import FourierSpace
from solver.integrate import Result, integrate, step_sizes
from solver.schemes.registry import scheme_info
from experiments.controllers import controller_settings, make_controller
from experiments.records import PROJECT, begin_run, finish_run, provenance, reusable_run, save_result
from .records import load_periodic
from .problems import SCHEMES, build_model, scheme

DEFAULTS: dict[str, Any] = {
    'case': 'mms', 'scheme': 'sdirk3_mrsav', 'nx': 32, 'ny': 32,
    'lx': 2*np.pi, 'ly': 2*np.pi, 'nu': .1, 'gamma': 1., 'r0': 0.,
    'amplitude': 1., 'epsilon': .1, 'k_f': 4, 'T': .1, 'dt': .01,
    'mode': 'fixed', 'initial_step': .01, 'controller': None,
    'scalar_atol': None, 'snapshots': None, 'stokes_tolerance': 1e-10,
    'log_every': 100,
    'boundary': 'periodic_xy', 'discretization': 'fourier_leray',
    'field_layout': 'collocated_yx', 'fft_norm': 'forward', 'dealias': 'strict_2_3',
}


def effective_config(config: dict) -> dict:
    unknown = set(config)-set(DEFAULTS)-{'steps'}
    if unknown:
        raise ValueError(f'Unknown periodic configuration keys: {sorted(unknown)}')
    cfg = {**DEFAULTS, **config}
    s = FourierSpace(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    for key in ('boundary', 'discretization', 'field_layout', 'fft_norm', 'dealias'):
        if cfg[key] != DEFAULTS[key]:
            raise ValueError(f'Periodic solver requires {key}={DEFAULTS[key]}')
    if cfg['case'] not in ('mms', 'kolmogorov') or cfg['scheme'] not in SCHEMES:
        raise ValueError('Unknown periodic case or scheme')
    if type(cfg['log_every']) is not int or cfg['log_every'] < 1:
        raise ValueError('log_every must be a positive integer')
    if not np.isfinite([cfg[k] for k in ('nu', 'T', 'gamma', 'r0', 'amplitude', 'epsilon', 'stokes_tolerance')]).all() \
            or min(cfg['nu'], cfg['T'], cfg['stokes_tolerance']) <= 0 or cfg['gamma'] < 0:
        raise ValueError('Invalid periodic physical parameters')
    if cfg['case'] == 'mms' and cfg['r0'] != 0:
        raise ValueError('Manufactured reference requires r0=0')
    if type(cfg['k_f']) is not int or cfg['k_f'] < 1:
        raise ValueError('k_f must be a positive integer')
    if cfg['case'] == 'kolmogorov' and cfg['k_f'] >= s.ny/3:
        raise ValueError('Kolmogorov forcing is outside the retained band')
    if cfg['mode'] == 'fixed':
        if cfg['controller'] is not None or cfg['scalar_atol'] is not None:
            raise ValueError('Controller options require adaptive mode')
        if 'steps' in config and 'dt' in config:
            raise ValueError('Use dt or steps, not both')
        if 'steps' in config:
            cfg['dt'] = None
        cfg['actual_steps'] = step_sizes(cfg['T'], dt=cfg['dt'], steps=cfg.get('steps')).tolist()
    elif cfg['mode'] == 'adaptive':
        if 'steps' in config or 'dt' in config:
            raise ValueError('Adaptive mode uses initial_step instead of dt/steps')
        options = {'kind': 'PI', 'atol': 1e-8, 'rtol': 5e-5, **(cfg['controller'] or {})}
        controller = make_controller(options, cfg['scheme'])
        if not np.isfinite(cfg['initial_step']) or cfg['initial_step'] < controller.min_step:
            raise ValueError('Invalid adaptive initial_step')
        cfg['controller'] = controller_settings(controller)
        cfg['dt'], cfg['actual_steps'] = None, None
        if cfg['scalar_atol'] is not None:
            if cfg['scheme'] != 'sdirk3_mrsav':
                raise ValueError('Joint scalar control is supported for sdirk3_mrsav')
            ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
    else:
        raise ValueError('mode must be fixed or adaptive')
    requests = [0., cfg['T']] if cfg['snapshots'] is None else cfg['snapshots']
    if any(not np.isfinite(t) or t < 0 or t > cfg['T'] for t in requests):
        raise ValueError('Snapshot outside periodic integration interval')
    cfg.update(s.metadata(), snapshots=list(requests),
               pressure_semantics=scheme_info(cfg['scheme']).pressure_semantics,
               initial_condition='analytic_periodic_streamfunction',
               forcing='continuous_nonlinear_mms' if cfg['case'] == 'mms' else '(-amplitude*sin(k_f*2*pi*y/ly),0)')
    return cfg


def run(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    cfg = effective_config(config)
    source = provenance()
    parent = Path(root)/'runs'/'periodic_ns'
    reused = reusable_run(parent, cfg, source, reader=lambda path: load_periodic(path, require_complete=True), rerun=rerun)
    if reused is not None:
        return reused
    directory, manifest = begin_run(parent, cfg, source, metadata={'experiment': 'periodic_ns'},
                                    started_key='started')
    (directory/'run.log').write_text(f'Started {manifest["started"]}\n{json.dumps(cfg, indent=2)}\n')
    def progress(count: int, t: float, elapsed: float) -> None:
        if count % cfg['log_every'] == 0 or abs(t-cfg['T']) < 1e-12:
            with (directory/'run.log').open('a') as stream:
                stream.write(f'accepted={count} t={t:.17g} wall={elapsed:.3f}s\n')
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
                                    snapshots=cfg['snapshots'], scalar_control=scalar, progress=progress)
    else:
        result = integrate(model, selected, initial, cfg['actual_steps'], snapshots=cfg['snapshots'], progress=progress)
    metrics = {'setup_cpu_seconds': setup_cpu, 'setup_wall_seconds': setup_wall,
               'integration_cpu_seconds': process_time()-cpu, 'integration_wall_seconds': result.seconds,
               'accepted_steps': len(result.times)-1, 'stokes_rhs_count': model.backend.rhs_count,
               'rejected_steps': sum(not a['accepted'] for a in result.attempts) if isinstance(result, AdaptiveResult) else 0}
    pending = directory/'results.pending.h5'
    info = scheme_info(cfg['scheme'])
    save_result(pending, result, metrics, n_stages=info.n_stages, n_roots=info.n_roots)
    with h5py.File(pending, 'a') as data:
        for key in ('boundary', 'discretization', 'field_layout', 'fft_norm', 'dealias'):
            data.attrs[key] = cfg[key]
        data.attrs['pressure_semantics'] = cfg['pressure_semantics']
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
    _, manifest = load_periodic(directory)
    if manifest['status'] != 'complete':
        raise SystemExit(f"Run failed: {manifest['error']}")


if __name__ == '__main__':
    main()

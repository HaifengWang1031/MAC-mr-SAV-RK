"""Run periodic NS; preserve the existing no-slip workflow and result schema."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time
from typing import Any
from uuid import uuid4

import h5py
import numpy as np

from solver.adaptivity import AdaptiveResult, IController, PIController, ScalarErrorControl, integrate_adaptive
from solver.fourier import FourierSpace
from solver.integrate import Result, integrate, step_sizes
from solver.schemes.registry import scheme_info
from experiments.workflow import PROJECT, digest, load_record, provenance, save_result, write_json
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


def make_controller(config: dict, name: str) -> IController:
    options = dict(config)
    kind = options.pop('kind', 'PI')
    if kind not in ('I', 'PI'):
        raise ValueError('Controller kind must be I or PI')
    order = scheme_info(name).estimator_order
    if order is None:
        raise ValueError('IMEX Euler requires a separate step-doubling estimator; use fixed mode')
    if 'estimator_order' in options and options['estimator_order'] != order:
        raise ValueError('Estimator order must match the embedded scheme')
    options['estimator_order'] = order
    if options.get('max_step', 1.) is None:
        options['max_step'] = float('inf')
    try:
        return (IController if kind == 'I' else PIController)(**options)
    except TypeError as error:
        raise ValueError(f'Invalid controller configuration: {error}') from error


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
        values = asdict(controller)
        values['max_step'] = None if np.isinf(controller.max_step) else controller.max_step
        cfg['controller'] = {'kind': options['kind'], **values}
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


def load_periodic(directory: Path, *, require_complete: bool = False) -> tuple[dict, dict]:
    cfg, manifest = load_record(directory, require_complete=require_complete)
    if manifest.get('experiment') != 'periodic_ns':
        raise ValueError('Expected a periodic_ns record')
    s = FourierSpace(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    info = scheme_info(cfg['scheme'])
    for key, value in s.metadata().items():
        if cfg.get(key) != value:
            raise ValueError(f'Periodic representation mismatch: {key}')
    with h5py.File(directory/'results.h5', 'r') as data:
        if data.attrs.get('status') != manifest['status']:
            raise ValueError('Periodic status mismatch')
        for key in ('boundary', 'discretization', 'field_layout', 'fft_norm', 'dealias'):
            if data.attrs.get(key) != cfg[key]:
                raise ValueError(f'HDF5 periodic representation mismatch: {key}')
        if data.attrs.get('pressure_semantics') != cfg.get('pressure_semantics') \
                or cfg.get('pressure_semantics') != info.pressure_semantics:
            raise ValueError('Periodic pressure semantics mismatch')
        if 'final/stage_pressure' in data:
            pressure = data['final/stage_pressure'][:]
            if pressure.shape != (info.n_stages, *s.shape) or not np.isfinite(pressure).all():
                raise ValueError('Invalid periodic stage pressure')
        for component in ('u', 'v'):
            if data['final/'+component].shape != s.shape \
                    or data['final/'+component].dtype.kind != 'f':
                raise ValueError('Periodic field layout/dtype mismatch')
    return cfg, manifest


def run(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    cfg = effective_config(config)
    source = provenance()
    identity = hashlib.sha256(json.dumps({'config': cfg, 'source': source}, sort_keys=True).encode()).hexdigest()
    parent = root/'runs'/'periodic_ns'
    parent.mkdir(parents=True, exist_ok=True)
    if not rerun:
        for directory in sorted(parent.iterdir()):
            if not directory.is_dir():
                continue
            try:
                old, manifest = load_periodic(directory, require_complete=True)
                if manifest.get('identity') == identity and old == cfg and manifest.get('source') == source:
                    return directory
            except (OSError, KeyError, ValueError):
                continue
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    directory = parent/f'{stamp}-{identity[:8]}-{uuid4().hex[:8]}'
    directory.mkdir()
    write_json(directory/'config.json', cfg)
    manifest = {'experiment': 'periodic_ns', 'run_id': directory.name, 'identity': identity,
                'source': source, 'status': 'running', 'started': datetime.now(timezone.utc).isoformat(),
                'config_sha256': digest(directory/'config.json')}
    write_json(directory/'manifest.json', manifest)
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
    manifest.update(status=result.status, error=result.error, metrics=metrics,
                    finished=datetime.now(timezone.utc).isoformat(), results_sha256=digest(directory/'results.h5'))
    write_json(directory/'manifest.json', manifest)
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

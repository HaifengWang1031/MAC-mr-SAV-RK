"""Compute, finalize, verify and reuse independent FEM records."""
import argparse
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter, process_time
from uuid import uuid4

import h5py
import numpy as np

from experiments.workflow import PROJECT, digest, load_record, provenance, save_result, write_json
from experiments.fem_backends import FEM_BACKEND_KEYS, STOKES_BACKENDS, validate_solver_options, solver_provenance
from experiments.periodic_ns.run import make_controller
from solver.adaptivity import AdaptiveResult, ScalarErrorControl, integrate_adaptive
from solver.fem import Space
from solver.integrate import Result, integrate, step_sizes
from solver.schemes.registry import make_scheme, scheme_info
from .problems import build_model
from .representation import METADATA, save_representation, validate
from .attempts import save_attempts

DEFAULTS = {'nx': 8, 'ny': 4, 'lx': 2., 'ly': 1., 'quadrature_order': 8,
            'nu': .2, 'amplitude': .1, 'T': .02, 'dt': .01, 'scheme': 'sdirk2_mrsav',
            'gamma': 1., 'r0': 0., 'cache_size': 4, 'stokes_tolerance': 1e-9, 'snapshots': None}

OPTIONS = ('mode', 'initial_step', 'controller', 'scalar_atol', 'stokes_backend', 'stokes_ordering',
           'stokes_factor_backend', 'stokes_threads')+FEM_BACKEND_KEYS


def effective_config(config: dict) -> dict:
    if set(config)-set(DEFAULTS)-{'steps'}-set(OPTIONS):
        raise ValueError('Unknown FEM configuration key')
    cfg = {**DEFAULTS, **config}
    Space(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'], cfg['quadrature_order'])
    if cfg['nx'] == cfg['ny'] == 1:
        raise ValueError('Mesh has too few free velocity DOFs for pressure constraints')
    info = scheme_info(cfg['scheme'])
    for key in ('nu', 'amplitude', 'T', 'stokes_tolerance'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid FEM {key}')
    if not np.isfinite(cfg['gamma']) or cfg['gamma'] < 0 or cfg['r0'] != 0.:
        raise ValueError('FEM MMS requires gamma >= 0 and r0=0')
    if type(cfg['cache_size']) is not int or cfg['cache_size'] < 1:
        raise ValueError('Invalid FEM cache_size')
    mode, backend = cfg.pop('mode', 'fixed'), cfg.get('stokes_backend', 'direct')
    if backend not in STOKES_BACKENDS:
        raise ValueError('Unknown FEM Stokes backend')
    validate_solver_options(cfg)
    if cfg.get('stokes_ordering', 'auto') not in ('auto', 'mmd', 'metis'):
        raise ValueError('Unknown FEM Stokes ordering')
    factor, threads = cfg.get('stokes_factor_backend', 'superlu'), cfg.get('stokes_threads')
    if factor not in ('superlu', 'pardiso') or (factor == 'pardiso' and cfg.get('stokes_ordering', 'auto') != 'auto') \
            or (threads is not None and (type(threads) is not int or threads < 1 or factor != 'pardiso')):
        raise ValueError('Invalid FEM factor backend/thread controls')
    if mode == 'fixed':
        if any(cfg.get(key) is not None for key in ('initial_step', 'controller', 'scalar_atol')):
            raise ValueError('Adaptive FEM options require mode=adaptive')
        for key in OPTIONS:
            cfg.pop(key, None)
        if backend != 'direct':
            cfg['stokes_backend'] = backend
        for key in ('stokes_ordering', 'stokes_factor_backend', 'stokes_threads')+FEM_BACKEND_KEYS:
            if key in config:
                cfg[key] = config[key]
        if 'steps' in config:
            if 'dt' in config:
                raise ValueError('Use dt or steps, not both')
            cfg['dt'] = None
        cfg['actual_steps'] = step_sizes(cfg['T'], dt=cfg['dt'], steps=cfg.get('steps')).tolist()
    elif mode == 'adaptive':
        if 'steps' in config or ('dt' in config and config['dt'] is not None):
            raise ValueError('Adaptive FEM uses initial_step, not dt/steps')
        options = {'kind': 'PI', 'atol': 1e-9, 'rtol': 1e-4, 'max_step': .02,
                   'min_factor': .5, 'max_factor': 1.5, **(cfg.get('controller') or {})}
        controller = make_controller(options, cfg['scheme'])
        initial_step = cfg.get('initial_step', .001)
        if isinstance(initial_step, bool) or not isinstance(initial_step, (int, float)) \
                or not np.isfinite(initial_step) or initial_step < controller.min_step:
            raise ValueError('Invalid FEM adaptive initial_step')
        values = asdict(controller)
        values['max_step'] = None if np.isinf(controller.max_step) else controller.max_step
        cfg.update(mode='adaptive', controller={'kind': options['kind'], **values}, initial_step=initial_step,
                   scalar_atol=cfg.get('scalar_atol'), stokes_backend=backend, actual_steps=None, dt=None)
        if cfg['scalar_atol'] is not None:
            if not cfg['scheme'].endswith('_mrsav'):
                raise ValueError('FEM scalar control requires an SAV scheme')
            ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
    else:
        raise ValueError('FEM mode must be fixed or adaptive')
    cfg['snapshots'] = [0., cfg['T']] if cfg['snapshots'] is None else list(cfg['snapshots'])
    if any(not np.isfinite(t) or t < 0 or t > cfg['T'] for t in cfg['snapshots']):
        raise ValueError('Invalid FEM snapshot times')
    cfg.update(METADATA, case='continuous_polynomial_mms', pressure_semantics=info.pressure_semantics,
               initial_condition='consistent_L2_then_weak_Leray_projection',
               forcing='analytic_ut_plus_advection_minus_nu_laplacian_plus_pressure_gradient')
    return cfg


def _identity(cfg: dict, source: dict) -> str:
    return hashlib.sha256(json.dumps({'config': cfg, 'source': source}, sort_keys=True).encode()).hexdigest()


def load_fem(directory: Path, *, require_complete: bool = False) -> tuple[dict, dict]:
    """Integrity and representation gate, also used by every readonly consumer."""
    directory = Path(directory)
    cfg, manifest = load_record(directory, require_complete=require_complete)
    if manifest.get('experiment') != 'fem_ns' or manifest.get('run_id') != directory.name:
        raise ValueError('Expected FEM run identity')
    source = manifest.get('source', {})
    packages = ('numpy', 'scipy', 'numba', 'h5py', 'sympy', 'matplotlib', 'scikit-fem')
    threads = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')
    if not isinstance(source, dict) or any(not isinstance(source.get(key), str) or not source[key]
                                            for key in ('code_sha256', 'python', 'platform')) \
            or len(source['code_sha256']) != 64 \
            or any(c not in '0123456789abcdef' for c in source['code_sha256']) \
            or not isinstance(source.get('packages'), dict) \
            or any(not isinstance(source['packages'].get(key), str) or not source['packages'][key] for key in packages) \
            or not isinstance(source.get('threads'), dict) \
            or any(key not in source['threads'] or (source['threads'][key] is not None
                   and not isinstance(source['threads'][key], str)) for key in threads):
        raise ValueError('Missing or malformed FEM provenance')
    try:
        inputs = {key: cfg[key] for key in DEFAULTS if key != 'dt' or 'steps' not in cfg}
        if 'steps' in cfg:
            inputs['steps'] = cfg['steps']
        inputs.update({key: cfg[key] for key in OPTIONS if key in cfg})
        if effective_config(inputs) != cfg or _identity(cfg, manifest['source']) != manifest['identity']:
            raise ValueError('FEM effective config/source identity mismatch')
        with h5py.File(directory/'results.h5', 'r') as data:
            validate(data, cfg, manifest)
    except (KeyError, TypeError) as error:
        raise ValueError(f'Missing FEM identity/metadata: {error}') from error
    return cfg, manifest


def run(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    cfg = effective_config(config)
    source = provenance()
    solver_provenance(source, cfg)
    source['packages']['scikit-fem'] = version('scikit-fem')
    source['packages']['pymetis'] = version('pymetis')
    if cfg.get('stokes_factor_backend') == 'pardiso':
        source['packages'].update({name: version(name) for name in ('pypardiso', 'mkl')})
    identity = _identity(cfg, source)
    parent = Path(root)/'runs'/'fem_ns'
    parent.mkdir(parents=True, exist_ok=True)
    if not rerun:
        for directory in sorted(parent.iterdir()):
            if not directory.is_dir():
                continue
            try:
                old, saved = load_fem(directory, require_complete=True)
                if old == cfg and saved['identity'] == identity and saved['source'] == source:
                    return directory
            except (OSError, KeyError, ValueError):
                continue
    directory = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                        +'-'+identity[:8]+'-'+uuid4().hex[:8])
    directory.mkdir()
    write_json(directory/'config.json', cfg)
    manifest: dict = {'experiment': 'fem_ns', 'run_id': directory.name, 'identity': identity,
                'source': source, 'status': 'running', 'rerun': rerun,
                'started': datetime.now(timezone.utc).isoformat(),
                'config_sha256': digest(directory/'config.json')}
    write_json(directory/'manifest.json', manifest)
    (directory/'run.log').write_text(f'Started {manifest["started"]}\n')
    cpu, wall = process_time(), perf_counter()
    try:
        model, initial = build_model(cfg)
        setup_cpu, setup_wall = process_time()-cpu, perf_counter()-wall
        cpu = process_time()
        selected = make_scheme(cfg['scheme'], cfg['gamma'])
        result: Result
        if cfg.get('mode') == 'adaptive':
            scalar = None if cfg['scalar_atol'] is None else ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
            result = integrate_adaptive(model, selected, initial, cfg['T'], cfg['initial_step'],
                                        controller=make_controller(cfg['controller'], cfg['scheme']),
                                        scalar_control=scalar, snapshots=cfg['snapshots'])
        else:
            result = integrate(model, selected, initial, cfg['actual_steps'], snapshots=cfg['snapshots'])
        metrics = {**model.performance_metrics(), 'setup_cpu_seconds': setup_cpu, 'setup_wall_seconds': setup_wall,
                   'integration_cpu_seconds': process_time()-cpu, 'integration_wall_seconds': result.seconds,
                   'accepted_steps': len(result.times)-1,
                   'rejected_steps': sum(not a['accepted'] for a in result.attempts) if isinstance(result, AdaptiveResult) else 0}
        pending = directory/'results.pending.h5'
        info = scheme_info(cfg['scheme'])
        save_result(pending, result, metrics, n_stages=info.n_stages, n_roots=info.n_roots,
                    pressure_semantics=info.pressure_semantics)
        with h5py.File(pending, 'a') as data:
            save_representation(data, model.velocity_basis, model.pressure_basis)
            if isinstance(result, AdaptiveResult):
                save_attempts(data, result.attempts)
        pending.replace(directory/'results.h5')
        manifest.update(status=result.status, error=result.error, metrics=metrics,
                        finished=datetime.now(timezone.utc).isoformat(), results_sha256=digest(directory/'results.h5'))
        write_json(directory/'manifest.json', manifest)
        with (directory/'run.log').open('a') as stream:
            stream.write(f'{result.status}: {result.error}\n{json.dumps(metrics, indent=2)}\n')
    except Exception as error:
        # Setup/storage failures cannot supply an accepted prefix; retain an auditable attempt.
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                        finished=datetime.now(timezone.utc).isoformat())
        write_json(directory/'manifest.json', manifest)
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
    _, manifest = load_fem(directory)
    if manifest['status'] != 'complete':
        raise SystemExit(f"Run failed: {manifest['error']}")


if __name__ == '__main__':
    main()

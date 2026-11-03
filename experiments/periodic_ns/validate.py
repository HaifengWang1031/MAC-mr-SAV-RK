"""Explicit reproducible validation and component timing, separate from analysis.

Temporal study fixes resolved finite-modal MMS and varies h; spatial study uses
infinite-modal steady Stokes, with no temporal error. The benchmark uses the same
MMS across sizes/schemes/schedules. It measures this periodic implementation only:
different-boundary no-slip MAC timings cannot establish a Fourier/MAC speedup.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import process_time
from typing import Any
from uuid import uuid4

import numpy as np

from solver.fourier import FourierModel, FourierSpace
from solver.schemes.registry import DEFAULT_SCHEMES
from experiments.records import PROJECT, digest, provenance, write_json
from .analyze import read_fields
from .problems import Manufactured, SCHEMES, build_model, smooth_stokes
from .run import effective_config, load_periodic, run

DEFAULTS: dict[str, Any] = {
    'temporal_size': 24, 'temporal_T': .4, 'temporal_counts': [10, 20, 40],
    'schemes': list(DEFAULT_SCHEMES), 'gammas': [1., 1000.], 'spatial_sizes': [12, 18, 24, 32],
    'benchmark_sizes': [32, 64, 128, 256], 'benchmark_T': .04, 'benchmark_steps': 10,
    'repeats': 3, 'operator_repeats': 10,
}


def validate(config: dict | None = None, *, root: Path = PROJECT) -> Path:
    supplied = config or {}
    if set(supplied)-set(DEFAULTS):
        raise ValueError('Unknown validation configuration')
    cfg = {**DEFAULTS, **supplied}
    if not cfg['schemes'] or any(name not in SCHEMES for name in cfg['schemes']):
        raise ValueError('Invalid validation schemes')
    for key in ('repeats', 'operator_repeats', 'benchmark_steps', 'temporal_size'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'Invalid {key}')
    for key in ('temporal_counts', 'spatial_sizes', 'benchmark_sizes'):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]) or any(type(n) is not int or n < 1 for n in cfg[key]):
            raise ValueError(f'Invalid {key}')
    if len(cfg['temporal_counts']) < 2 or cfg['temporal_counts'] != sorted(cfg['temporal_counts']):
        raise ValueError('Temporal counts must be strictly increasing')
    if cfg['benchmark_steps'] % 2 or not cfg['gammas'] or any(not np.isfinite(g) or g < 0 for g in cfg['gammas']) \
            or not np.isfinite([cfg['temporal_T'], cfg['benchmark_T']]).all() \
            or min(cfg['temporal_T'], cfg['benchmark_T']) <= 0:
        raise ValueError('Invalid validation time/gamma parameters')
    for n in [cfg['temporal_size'], *cfg['spatial_sizes'], *cfg['benchmark_sizes']]:
        FourierSpace(n, n)
    directory = root/'runs'/'periodic_ns_validation'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    write_json(directory/'config.json', cfg)
    source = provenance()
    manifest = {'status': 'running', 'source': source, 'id': directory.name,
                'experiment': 'periodic_ns_validation',
                'config_sha256': digest(directory/'config.json')}
    write_json(directory/'manifest.json', manifest)
    temporal: list[dict[str, Any]] = []
    spatial: list[dict[str, Any]] = []
    benchmark: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    results = {'temporal': temporal, 'spatial': spatial, 'benchmark': benchmark, 'profiles': profiles,
               'limits': ['CPU timings refer to periodic Fourier only; no periodic MAC baseline',
                          'short-time MMS and steady Stokes validation, not a long-time PDE proof',
                          'scalar errors are recorded separately; no high-order scalar claim']}
    try:
        for gamma in cfg['gammas']:
            for name in cfg['schemes']:
                errors, r_errors, members = [], [], []
                for count in cfg['temporal_counts']:
                    path = run({'case': 'mms', 'scheme': name, 'nx': cfg['temporal_size'],
                                'ny': cfg['temporal_size'], 'nu': .1, 'gamma': gamma,
                                'T': cfg['temporal_T'], 'dt': cfg['temporal_T']/count}, root=directory)
                    load_periodic(path, require_complete=True)
                    _, fields = read_fields(path)
                    s = FourierSpace(cfg['temporal_size'], cfg['temporal_size'])
                    delta = np.stack([fields['u'], fields['v']])-Manufactured(s, .1).exact(fields['final_time'])
                    errors.append(float(np.sqrt(s.area*np.mean(np.sum(delta**2, axis=0)))))
                    r_errors.append(abs(fields['r']))
                    members.append(str(path.relative_to(directory)))
                counts = np.array(cfg['temporal_counts'])
                rates = (np.log(np.array(errors[:-1])/errors[1:])/np.log(counts[1:]/counts[:-1])).tolist()
                temporal.append({'scheme': name, 'gamma': gamma, 'velocity_L2_errors': errors,
                                 'velocity_rates': rates, 'scalar_absolute_errors': r_errors, 'members': members})
        for n in cfg['spatial_sizes']:
            s = FourierSpace(n, n)
            exact, force = smooth_stokes(s)
            model = FourierModel(s, 1.)
            solved = model.solve(s.to_spectral(force), mass=0., viscosity=1.)
            delta = s.to_physical(solved.velocity)-exact
            error = float(np.sqrt(s.area*np.mean(np.sum(delta**2, axis=0))))
            # Preserve exact and computed fields so spatial errors can be re-evaluated.
            np.savez_compressed(directory/f'stokes_{n}.npz', exact=exact,
                                velocity=s.to_physical(solved.velocity), pressure=s.to_physical(solved.pressure))
            spatial.append({'size': n, 'velocity_L2_error': error, 'residual': solved.residual,
                            'divergence_inf': solved.divergence_inf, 'space': s.metadata()})
        for repeat in range(cfg['repeats']):
            for n in cfg['benchmark_sizes']:
                # Warm the FFT/solve path before measuring components, no JIT needed.
                base = effective_config({'nx': n, 'ny': n})
                model, initial, _ = build_model(base)
                z = model.vector(initial)
                model.nonlinear(z)
                model.solve(z, mass=1., viscosity=.001)
                for label, operation in (
                    ('nonlinear', lambda: model.nonlinear(z)),
                    ('stokes_single_rhs', lambda: model.solve(z, mass=1., viscosity=.001)),
                    ('stokes_two_rhs', lambda: model.solve_columns([z, z], mass=1., viscosity=.001)),
                    ('diagnostics', lambda: model.diagnostics(initial))):
                    cpu = process_time()
                    for _ in range(cfg['operator_repeats']):
                        operation()
                    profiles.append({'size': n, 'repeat': repeat, 'component': label,
                                     'cpu_seconds_per_call': (process_time()-cpu)/cfg['operator_repeats']})
                h = cfg['benchmark_T']/cfg['benchmark_steps']
                for name in cfg['schemes']:
                    for schedule in ('fixed', 'variable'):
                        steps = [h]*cfg['benchmark_steps'] if schedule == 'fixed' else [h*.5, h*1.5]*(cfg['benchmark_steps']//2)
                        path = run({'nx': n, 'ny': n, 'scheme': name, 'gamma': 1., 'T': cfg['benchmark_T'],
                                    'steps': steps, 'snapshots': []}, root=directory, rerun=True)
                        _, saved = load_periodic(path, require_complete=True)
                        _, fields = read_fields(path)
                        benchmark.append({'size': n, 'scheme': name, 'schedule': schedule, 'repeat': repeat,
                                          'member': str(path.relative_to(directory)), **saved['metrics'],
                                          'max_stage_residual': fields['max_stage_residual'],
                                          'max_physical_divergence': float(max(fields['diagnostics']['divergence_inf']))})
                print(f'benchmark repeat={repeat+1}/{cfg["repeats"]} grid={n}', flush=True)
                write_json(directory/'results.json', results)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(directory/'results.json', results)
        manifest['artifacts'] = {path.name: digest(path) for path in directory.iterdir()
                                 if path.name == 'results.json' or path.suffix == '.npz'}
        write_json(directory/'manifest.json', manifest)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(validate(None if args.config is None else json.loads(args.config.read_text()), root=args.root))


if __name__ == '__main__':
    main()

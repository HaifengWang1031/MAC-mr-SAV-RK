"""Short reproducible channel verification; saved evidence, separate read-only analysis.

Temporal MMS fixes the MAC grid and uses independent NumPy loads. Spatial Stokes
uses continuous derivatives and no time discretisation. Neither study establishes
long-time turbulent accuracy, global scalar control or physical endpoint-pressure order.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from solver.channel import ChannelGrid, ChannelModel
from solver.schemes.registry import DEFAULT_SCHEMES, scheme_info
from experiments.records import PROJECT, digest, provenance, write_json
from .analyze import read_fields, velocity_error
from .problems import Manufactured, SCHEMES
from .run import load_channel, run

DEFAULTS: dict[str, Any] = {
    'temporal_nx': 12, 'temporal_ny': 10, 'temporal_T': .4,
    'temporal_counts': [40, 80, 160], 'schemes': list(DEFAULT_SCHEMES), 'gammas': [1., 1000.],
    'spatial_sizes': [8, 16, 32], 'smoke_cases': ['startup', 'perturbed'],
}


def validate(config: dict | None = None, *, root: Path = PROJECT) -> Path:
    supplied = config or {}
    if set(supplied)-set(DEFAULTS): raise ValueError('Unknown channel validation configuration')
    cfg = {**DEFAULTS, **supplied}
    ChannelGrid(cfg['temporal_nx'], cfg['temporal_ny'])
    if not cfg['schemes'] or len(set(cfg['schemes'])) != len(cfg['schemes']) \
            or any(name not in SCHEMES for name in cfg['schemes']):
        raise ValueError('Invalid validation schemes')
    if not cfg['gammas'] or len(set(cfg['gammas'])) != len(cfg['gammas']) \
            or any(not np.isfinite(g) or g < 0 for g in cfg['gammas']) \
            or not np.isfinite(cfg['temporal_T']) or cfg['temporal_T'] <= 0:
        raise ValueError('Invalid validation gamma/time')
    for key in ('temporal_counts', 'spatial_sizes'):
        values = cfg[key]
        if len(values) < 2 or any(type(n) is not int or n < 3 for n in values) \
                or any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f'{key} must be strictly increasing integers >=3')
    if len(set(cfg['smoke_cases'])) != len(cfg['smoke_cases']) \
            or any(case not in ('startup', 'perturbed') for case in cfg['smoke_cases']):
        raise ValueError('Invalid smoke cases')
    directory = root/'runs'/'channel_ns_validation'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    write_json(directory/'config.json', cfg)
    manifest = {'experiment': 'channel_ns_validation', 'status': 'running', 'source': provenance(),
                'config_sha256': digest(directory/'config.json'), 'id': directory.name}
    write_json(directory/'manifest.json', manifest)
    temporal: list[dict[str, Any]] = []
    spatial: list[dict[str, Any]] = []
    smoke: list[dict[str, Any]] = []
    results = {'temporal': temporal, 'spatial': spatial, 'smoke': smoke,
               'limits': ['finite-time verification, not a long-time turbulence validation',
                          'scalar errors reported separately; no high-order scalar claim',
                          'step/stage pressure semantics depend on the selected scheme; no pressure-order claim',
                          'temporal/spatial study uses direct backend; FFT has same-discretisation tests',
                          'wall shear uses half-cell gradient; no second-order shear claim']}
    (directory/'run.log').write_text('Channel temporal/spatial verification started\n')
    try:
        for gamma in cfg['gammas']:
            for name in cfg['schemes']:
                members, errors, scalars = [], [], []
                for count in cfg['temporal_counts']:
                    path = run({'case': 'discrete_mms', 'scheme': name, 'gamma': gamma,
                                'nx': cfg['temporal_nx'], 'ny': cfg['temporal_ny'],
                                'T': cfg['temporal_T'], 'dt': cfg['temporal_T']/count}, root=directory)
                    load_channel(path, require_complete=True)
                    child_cfg, fields = read_fields(path)
                    error = velocity_error(child_cfg, fields)
                    assert error is not None
                    errors.append(error[0]); scalars.append(abs(fields['r']))
                    members.append(str(path.relative_to(directory)))
                counts = np.array(cfg['temporal_counts'])
                rates = np.log(np.array(errors[:-1])/errors[1:])/np.log(counts[1:]/counts[:-1])
                temporal.append({'scheme': name, 'gamma': gamma, 'members': members,
                                 'velocity_L2_errors': errors, 'velocity_rates': rates.tolist(),
                                 'scalar_absolute_errors': scalars})
                with (directory/'run.log').open('a') as log:
                    log.write(f'{name} gamma={gamma}: velocity rates={rates.tolist()}\n')
        for n in cfg['spatial_sizes']:
            grid = ChannelGrid(2*n, n, 2*np.pi, 1.)
            reference = Manufactured(grid, 1.)
            model = ChannelModel(grid, 1.)
            sol = model.solve(reference.stiffness+reference.gradient, mass=0., viscosity=1.)
            u, v = grid.unpack(sol.velocity)
            eu, ev = grid.unpack(reference.base)
            filename = f'stokes_{n}.npz'
            np.savez_compressed(directory/filename, u=u, v=v, pressure=sol.pressure, exact_u=eu, exact_v=ev)
            spatial.append({'nx': grid.nx, 'ny': grid.ny, 'lx': grid.lx, 'ly': grid.ly,
                            'artifact': filename, 'velocity_L2_error': grid.norm(sol.velocity-reference.base),
                            'residual': sol.residual, 'divergence_inf': sol.divergence_inf})
        for case in cfg['smoke_cases']:
            for name in cfg['schemes']:
                path = run({'case': case, 'scheme': name, 'nx': cfg['temporal_nx'], 'ny': cfg['temporal_ny'],
                            'T': .05, 'dt': .005}, root=directory)
                _, saved = load_channel(path, require_complete=True)
                smoke.append({'case': case, 'scheme': name, 'member': str(path.relative_to(directory)),
                              'metrics': saved['metrics']})
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(directory/'results.json', results)
        manifest['artifacts'] = {path.name: digest(path) for path in directory.iterdir()
                                 if path.name == 'results.json' or path.suffix == '.npz'}
        write_json(directory/'manifest.json', manifest)
        with (directory/'run.log').open('a') as log:
            log.write(f"{manifest['status']}: {manifest.get('error', '')}\n")
    return directory


def analyze_validation(directory: Path, *, root: Path = PROJECT) -> Path:
    """Recompute errors from verified fields; no solver invocation."""
    directory = directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('experiment') != 'channel_ns_validation' or manifest.get('status') != 'complete':
        raise ValueError('Complete channel validation study required')
    if digest(directory/'config.json') != manifest['config_sha256']:
        raise ValueError('Validation config checksum mismatch')
    for name, checksum in manifest['artifacts'].items():
        if digest(directory/name) != checksum: raise ValueError('Validation artifact checksum mismatch')
    cfg = json.loads((directory/'config.json').read_text())
    results = json.loads((directory/'results.json').read_text())
    if len(results['temporal']) != len(cfg['schemes'])*len(cfg['gammas']) \
            or len(results['spatial']) != len(cfg['spatial_sizes']) \
            or len(results['smoke']) != len(cfg['schemes'])*len(cfg['smoke_cases']):
        raise ValueError('Incomplete channel validation matrix')
    for row in results['temporal']:
        if len(row['members']) != len(cfg['temporal_counts']): raise ValueError('Incomplete temporal row')
        errors, scalars = [], []
        for member, count in zip(row['members'], cfg['temporal_counts']):
            child = directory/member
            child_cfg, saved = load_channel(child, require_complete=True)
            if saved['source'] != manifest['source'] \
                    or (child_cfg['scheme'], child_cfg['gamma'], child_cfg['nx'], child_cfg['ny'], child_cfg['T']) != \
                       (row['scheme'], row['gamma'], cfg['temporal_nx'], cfg['temporal_ny'], cfg['temporal_T']) \
                    or len(child_cfg['actual_steps']) != count or child_cfg['case'] != 'discrete_mms':
                raise ValueError('Temporal member/source mismatch')
            _, fields = read_fields(child)
            error = velocity_error(child_cfg, fields)
            assert error is not None
            errors.append(error[0]); scalars.append(abs(fields['r']))
        if not np.allclose(errors, row['velocity_L2_errors'], rtol=1e-12, atol=1e-15) \
                or not np.allclose(scalars, row['scalar_absolute_errors'], rtol=1e-12, atol=1e-15):
            raise ValueError('Saved temporal error mismatch')
        counts = np.array(cfg['temporal_counts'])
        rates = np.log(np.array(errors[:-1])/errors[1:])/np.log(counts[1:]/counts[:-1])
        if not np.allclose(rates, row['velocity_rates'], rtol=1e-12, atol=1e-12):
            raise ValueError('Saved temporal rate mismatch')
    for row in results['spatial']:
        grid = ChannelGrid(row['nx'], row['ny'], row['lx'], row['ly'])
        with np.load(directory/row['artifact']) as stored:
            exact = grid.pack(stored['exact_u'], stored['exact_v'])
            np.testing.assert_allclose(exact, Manufactured(grid, 1.).base, rtol=1e-13, atol=1e-14)
            spatial_error = grid.norm(grid.pack(stored['u'], stored['v'])-exact)
        if not np.isclose(spatial_error, row['velocity_L2_error'], rtol=1e-12, atol=1e-15):
            raise ValueError('Saved spatial error mismatch')
    for row in results['smoke']:
        child_cfg, saved = load_channel(directory/row['member'], require_complete=True)
        if saved['source'] != manifest['source'] or saved['metrics'] != row['metrics'] \
                or (child_cfg['case'], child_cfg['scheme']) != (row['case'], row['scheme']):
            raise ValueError('Smoke member/source mismatch')
    spatial_errors = np.array([r['velocity_L2_error'] for r in results['spatial']])
    sizes = np.array(cfg['spatial_sizes'])
    spatial_rates = (np.log(spatial_errors[:-1]/spatial_errors[1:])/np.log(sizes[1:]/sizes[:-1])).tolist()
    expected = all(min(r['velocity_rates']) > scheme_info(r['scheme']).order-.1
                   for r in results['temporal']) and min(spatial_rates) > 1.8
    report = root/'reports'/'channel_ns_validation'/directory.name
    report.mkdir(parents=True, exist_ok=True)
    write_json(report/'analysis.json', {'run': str(directory), 'manifest_sha256': digest(directory/'manifest.json'),
                                       'analysis_source': provenance(), **results, 'spatial_rates': spatial_rates,
                                       'expected_velocity_orders_observed': expected})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(validate(None if args.config is None else json.loads(args.config.read_text()), root=args.root))


if __name__ == '__main__': main()

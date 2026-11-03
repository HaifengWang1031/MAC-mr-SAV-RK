"""Reproducible TS growth check followed by an illustrative nonlinear channel run.

Controls: fixed centreline speed/height and first x mode; vary dt, grid and Re.
Observables: perturbation amplitude, fitted growth, eigenvalue, full residuals.
The short linear checks qualify instability, not nonlinear solution convergence.
Computation and read-only analysis have separate CLI entry points.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from experiments.records import PROJECT, digest, provenance, write_json
from .animate import read_movie
from .run import load_channel, run
from .ts_mode import continuous_eigenvalue

DEFAULTS: dict[str, Any] = {
    'reynolds': 30000., 'alpha': .8, 'height': 2., 'centreline_speed': 1.,
    'nx': 96, 'ny': 192, 'fine_nx': 128, 'fine_ny': 256,
    'T': 400., 'dt': .05, 'epsilon': .025, 'snapshot_interval': 1.,
    'verification_T': 20., 'verification_epsilon': 1e-4, 'stable_reynolds': 1000.,
    'growth_relative_gate': .02,
}


def channel_config(config: dict[str, Any], *, verification: bool = False,
                   refinement: str = 'base') -> dict[str, Any]:
    """Derive viscosity and pressure gradient from Re, Umax and half-height."""
    c = config
    re = c['stable_reynolds'] if refinement == 'stable' else c['reynolds']
    nu = c['centreline_speed']*(c['height']/2)/re
    duration = c['verification_T'] if verification else c['T']
    interval = 1. if verification else c['snapshot_interval']
    count = round(duration/interval)
    if count < 1 or not np.isclose(count*interval, duration):
        raise ValueError('Duration must be an integer multiple of snapshot interval')
    return {'case': 'ts_wave', 'backend': 'fft', 'scheme': 'sdirk3',
            'nx': c['fine_nx'] if refinement == 'grid' else c['nx'],
            'ny': c['fine_ny'] if refinement == 'grid' else c['ny'],
            'lx': np.pi*c['height']/c['alpha'], 'ly': c['height'], 'nu': nu,
            'driving': 8*nu*c['centreline_speed']/c['height']**2,
            'epsilon': c['verification_epsilon'] if verification else c['epsilon'],
            'T': duration, 'dt': c['dt']/2 if refinement == 'time' else c['dt'],
            'snapshots': np.linspace(0., duration, count+1).tolist(), 'log_every': 500}


def growth_record(directory: Path) -> dict[str, Any]:
    """Fit saved small-perturbation amplitudes without building a model."""
    import h5py
    data = read_movie(directory)
    _, manifest = load_channel(directory, require_complete=True)
    with h5py.File(directory/'results.h5', 'r') as stored:
        metadata = dict(stored['initial_mode'].attrs)
    slope = float(np.polyfit(data.times, np.log(data.perturbation_norm), 1)[0])
    return {'run': str(directory.resolve()), 'run_id': directory.name,
            'results_sha256': manifest['results_sha256'],
            'growth_fit': slope, 'growth_discrete': float(metadata['eigenvalue_real']),
            'amplitude_gain': float(data.perturbation_norm[-1]/data.perturbation_norm[0]),
            'max_divergence': data.max_divergence, 'max_stage_residual': data.max_stage_residual}


def verify_checks(checks: dict[str, dict], reference: complex, relative_gate: float) -> dict:
    """Defined numerical gates for temporal and spatial instability evidence."""
    base, time, grid, stable = [checks[name] for name in ('base', 'time', 'grid', 'stable')]
    temporal = abs(base['growth_fit']-time['growth_fit'])/abs(time['growth_fit'])
    spatial = abs(base['growth_discrete']-grid['growth_discrete'])/abs(grid['growth_discrete'])
    continuous = abs(grid['growth_discrete']-reference.real)/abs(reference.real)
    fits = max(abs(row['growth_fit']-row['growth_discrete'])/abs(row['growth_discrete'])
               for row in checks.values())
    passed = (min(row['growth_fit'] for key, row in checks.items() if key != 'stable') > 0
              and stable['growth_fit'] < 0
              and max(temporal, spatial, continuous, fits) < relative_gate
              and max(row['max_divergence'] for row in checks.values()) < 1e-10
              and max(row['max_stage_residual'] for row in checks.values()) < 1e-8)
    return {'passed': bool(passed), 'time_relative_change': temporal, 'grid_relative_change': spatial,
            'fine_grid_vs_continuous_relative_change': continuous,
            'max_fit_vs_discrete_relative_difference': fits, 'relative_gate': relative_gate}


def run_study(config: dict[str, Any], *, root: Path = PROJECT) -> Path:
    """Compute checked small-amplitude runs, then the nonlinear illustration."""
    if set(config)-set(DEFAULTS):
        raise ValueError('Unknown instability configuration key')
    cfg = {**DEFAULTS, **config}
    if not np.isfinite(list(cfg.values())).all() or min(cfg.values()) <= 0:
        raise ValueError('Instability parameters must be finite and positive')
    directory = root/'runs'/'channel_instability'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                                                +'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    write_json(directory/'config.json', cfg)
    manifest: dict[str, Any] = {'status': 'running', 'source': provenance(), 'checks': {},
                              'config_sha256': digest(directory/'config.json')}
    write_json(directory/'manifest.json', manifest)
    log = directory/'run.log'
    log.write_text(f'Started {directory}\n')
    try:
        benchmark = continuous_eigenvalue(10000., 1.)
        known = .0037396706229799-.2375264888204682j
        if abs(benchmark-known) > 2e-8:
            raise RuntimeError('Independent Orr--Sommerfeld benchmark failed')
        reference = continuous_eigenvalue(cfg['reynolds'], cfg['alpha'], 100) \
            *cfg['centreline_speed']/(cfg['height']/2)
        write_json(directory/'reference.json', {'benchmark_real': benchmark.real,
            'benchmark_imag': benchmark.imag, 'benchmark_error': abs(benchmark-known),
            'continuous_growth': reference.real, 'continuous_frequency': reference.imag,
            'source': 'https://mpj1001.user.srcf.net/papers/AMR_Juniper_Hanifi_Theofilis_2014.pdf'})
        for name in ('base', 'time', 'grid', 'stable'):
            child = run(channel_config(cfg, verification=True, refinement=name), root=root)
            manifest['checks'][name] = growth_record(child)
            write_json(directory/'manifest.json', manifest)
            with log.open('a') as stream:
                stream.write(f'{name}: {json.dumps(manifest["checks"][name])}\n')
            print(f'{name}: growth={manifest["checks"][name]["growth_fit"]:.8g}', flush=True)
        gates = verify_checks(manifest['checks'], reference, cfg['growth_relative_gate'])
        manifest['gates'] = gates
        if not gates['passed']:
            raise RuntimeError(f'Instability qualification failed: {gates}')
        write_json(directory/'manifest.json', manifest)
        print(f'growth checks passed; starting nonlinear run in {directory}', flush=True)
        child = run(channel_config(cfg), root=root)
        manifest['nonlinear'] = growth_record(child)
        if manifest['nonlinear']['amplitude_gain'] < 2.:
            raise RuntimeError('Nonlinear illustration did not show at least twofold amplitude growth')
        manifest.update(status='complete', reference_sha256=digest(directory/'reference.json'))
        with log.open('a') as stream:
            stream.write(f'complete: {json.dumps(manifest["nonlinear"])}\n')
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        with log.open('a') as stream:
            stream.write(manifest['error']+'\n')
        raise
    finally:
        write_json(directory/'manifest.json', manifest)
    return directory


def analyze_study(directory: Path, *, root: Path = PROJECT) -> Path:
    """Recheck checksums and fit saved fields; no eigensolve or PDE integration."""
    cfg = json.loads((directory/'config.json').read_text())
    saved = json.loads((directory/'manifest.json').read_text())
    if saved['status'] != 'complete' or digest(directory/'config.json') != saved['config_sha256'] \
            or digest(directory/'reference.json') != saved['reference_sha256']:
        raise ValueError('Incomplete or changed instability study')
    reference = json.loads((directory/'reference.json').read_text())
    checks = {}
    for name, record in saved['checks'].items():
        checks[name] = growth_record(Path(record['run']))
        if checks[name]['results_sha256'] != record['results_sha256']:
            raise ValueError('Changed instability member')
    nonlinear = growth_record(Path(saved['nonlinear']['run']))
    if nonlinear['results_sha256'] != saved['nonlinear']['results_sha256']:
        raise ValueError('Changed nonlinear member')
    report = root/'reports'/'channel_instability'/directory.name
    report.mkdir(parents=True, exist_ok=True)
    write_json(report/'analysis.json', {'study': str(directory.resolve()), 'checks': checks,
        'reference': reference, 'nonlinear': nonlinear,
        'gates': verify_checks(checks, complex(reference['continuous_growth'], reference['continuous_frequency']),
                              cfg['growth_relative_gate']),
        'limits': 'Short linear refinement checks; nonlinear endpoint/animation is not convergence-certified'})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--analyze', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    if args.analyze is not None:
        print(analyze_study(args.analyze, root=args.root))
    else:
        config = {} if args.config is None else json.loads(args.config.read_text())
        print(run_study(config, root=args.root))


if __name__ == '__main__':
    main()

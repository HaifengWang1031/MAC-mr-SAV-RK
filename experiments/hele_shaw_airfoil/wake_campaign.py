"""Frozen-source high-inertia wake exploration, with explicit pilot/production phases.

One serial process per case keeps sparse factors out of subsequent cases.  Runs
and reports are separate; the readonly wake analysis never starts a solver.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from uuid import uuid4

import h5py
import numpy as np

from experiments.workflow import PROJECT, digest, provenance, write_json
from .config import effective_config


def configurations() -> dict[str, dict[str, Any]]:
    """Keep the geometry, nominal Darcy speed and pressure ends controlled."""
    base: dict[str, Any] = dict(angle=10., gap_ratio=.02, surface_points=120,
        layer_cells=4, far_size=.4, wake_region=[.5, 5., -.7, .5], wake_size=.025,
        scheme='sdirk3_mrsav', mode='adaptive', dt=None, initial_step=1e-4,
        controller=dict(kind='PI', atol=1e-9, rtol=1e-4, min_step=1e-7,
                        max_step=.005, safety=.9, min_factor=.5, max_factor=1.5,
                        max_rejections=16, max_steps=100000),
        scalar_atol=1e-4, cache_size=1, stokes_backend='recycled',
        ramp_time=1., T=20., snapshots=np.linspace(0., 20., 401).tolist(),
        log_every=50, darcy_reference=False, outlet_treatment='natural')
    low = {**base, 'nu': 1e-4, 'pressure_drop': 27.}
    high = {**base, 'nu': 1/30000, 'pressure_drop': 9.}
    pilot_low = {**low, 'T': .1, 'snapshots': [0., .05, .1], 'log_every': 10}
    pilot_high = {**high, 'T': .1, 'snapshots': [0., .05, .1], 'log_every': 10}
    return dict(pilot_direct={**pilot_low, 'stokes_backend': 'direct'},
        pilot_recycled=pilot_low, pilot_high=pilot_high,
        pilot_time={**pilot_high, 'controller': {**base['controller'], 'max_step': .0025}},
        pilot_mesh={**pilot_high, 'layer_cells': 5, 'wake_size': .02},
        re10000=low, re30000=high)


def prepare(root: Path = PROJECT, *, cases: dict[str, dict[str, Any]] | None = None,
            purpose: str | None = None) -> Path:
    """Snapshot the exact source and all requested inputs before any long run."""
    label = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory = root/'runs/hele_shaw_wake'/label
    source = directory/'source'
    source.mkdir(parents=True)
    files = [PROJECT/name for name in ('pyproject.toml', 'uv.lock', '.python-version')]
    for name in ('solver', 'experiments', 'tools'):
        files.extend(sorted((PROJECT/name).rglob('*.py')))
    hashes = {}
    for path in files:
        relative = path.relative_to(PROJECT)
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(target)
    write_json(directory/'source_manifest.json', hashes)
    report = root/'reports/hele_shaw_wake'/label
    custom = cases is not None
    cases = configurations() if cases is None else cases
    for name, config in cases.items():
        case = directory/'cases'/name
        case.mkdir(parents=True)
        write_json(case/'input.json', config)
        write_json(case/'effective_config.json', effective_config(config))
    write_json(directory/'request.json', dict(cases=cases, report=str(report.resolve()),
        source=provenance(), purpose=__doc__ if purpose is None else purpose,
        interpretation='2D MWR reduced-model high-inertia exploration; nominal Darcy Re '
            '10000/30000, actual Re measured from outlet flux. No promised separation, '
            'resolved shedding, grid convergence or 3D Hele-Shaw validity.',
        pilot_limits='T=0.1 startup tests qualify workflow/backend only, not mature high-Re accuracy.',
        guard='No forced minimum step or scalar clipping; numerical failure preserves accepted prefix.'))
    if custom:
        request = json.loads((directory/'request.json').read_text())
        request['interpretation'] = 'Explicit supplied case configs; 2D reduced-model exploration. No promised T=20 completion, shedding or 3D validity.'
        request['pilot_limits'] = 'Replay and finite-time pilot qualify the observed failure mechanism only; no mature-flow convergence claim.'
        write_json(directory/'request.json', request)
    (directory/'STATUS.md').write_text(
        '# High-inertia Hele-Shaw wake\n\n'
        'Read `state.json`, `driver.log`, and `cases/*/console.log` for program-owned progress.\n'
        'The active integration writes `cases/*/runs/hele_shaw_airfoil/*/run.log`.\n'
        'Reports are separate; no restart/checkpoint continuation is implemented.\n')
    return directory.resolve()


def verify_source(directory: Path) -> None:
    if PROJECT.resolve() != (directory/'source').resolve():
        raise ValueError('Execute the worker from the frozen source directory')
    expected = json.loads((directory/'source_manifest.json').read_text())
    current = {str(path.relative_to(PROJECT)): digest(path)
               for name in ('solver', 'experiments', 'tools')
               for path in (PROJECT/name).rglob('*.py')}
    current.update({name: digest(PROJECT/name) for name in ('pyproject.toml', 'uv.lock', '.python-version')})
    if current != expected or provenance() != json.loads((directory/'request.json').read_text())['source']:
        raise ValueError('Frozen source/environment differs from the prepared request')


def execute_case(directory: Path, name: str) -> dict[str, Any]:
    """Run one explicit config, validate the full saved record, then summarize."""
    from .records import load
    from .run import run
    verify_source(directory)
    case = directory/'cases'/name
    config = json.loads((case/'input.json').read_text())
    run_dir = run(config, root=case)
    cfg, manifest, *_ = load(run_dir, require_complete=False)
    summary: dict[str, Any] = dict(run=str(run_dir), run_id=run_dir.name,
        status=manifest['status'], final_time=manifest.get('final_time'),
        source=manifest['source'], results_sha256=manifest.get('results_sha256'),
        metrics=manifest.get('metrics'), error=manifest.get('error'))
    with h5py.File(run_dir/'results.h5', 'r') as data:
        summary.update(max_abs_r=float(np.max(abs(data['diagnostics/r'][:]))),
            max_stage_continuity=float(np.max(data['stages/continuity_residual'][:], initial=0.)),
            max_stage_residual=float(np.max(data['stages/residual'][:], initial=0.)),
            final_actual_Re=float(data['diagnostics/outlet_mean_velocity'][-1]*cfg['chord']/cfg['nu']))
    summary['diagnostic_qualification'] = (manifest['status'] == 'complete'
        and summary['final_time'] == cfg['T'] and summary['max_abs_r'] < .05
        and summary['max_stage_continuity'] <= cfg['stokes_tolerance']
        and summary['max_stage_residual'] <= cfg['stokes_tolerance'])
    summary['qualification_limits'] = 'r<0.05 is an experiment diagnostic, not a stability or accuracy theorem.'
    write_json(case/'result.json', summary)
    return summary


def drive(directory: Path, names: list[str], *, render: bool) -> None:
    """Persist progress and fail explicitly; no polling agent is needed."""
    verify_source(directory)
    request = json.loads((directory/'request.json').read_text())
    state: dict[str, Any] = dict(status='running', pid=os.getpid(), cases={}, started_utc=datetime.now(timezone.utc).isoformat())
    try:
        for name in names:
            state['active_case'] = name
            write_json(directory/'state.json', state)
            print('START', name, datetime.now(timezone.utc).isoformat(), flush=True)
            case = directory/'cases'/name
            with (case/'console.log').open('a', buffering=1) as log:
                subprocess.run([sys.executable, '-u', '-m', 'experiments.hele_shaw_airfoil.wake_campaign', 'case',
                    '--campaign', str(directory), '--name', name], cwd=PROJECT,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((case/'result.json').read_text())
            state['cases'][name] = summary
            write_json(directory/'state.json', state)
            if not summary['diagnostic_qualification']:
                raise RuntimeError(f'{name} failed the diagnostic qualification; inspect the saved prefix')
            if render:
                from .wake_analysis import analyze_wake
                state['phase'] = 'readonly_analysis'
                write_json(directory/'state.json', state)
                analyze_wake(Path(summary['run']), output=Path(request['report'])/name)
                state['phase'] = 'integration'
            print('DONE', name, summary['final_time'], flush=True)
        state.update(status='complete', active_case=None, finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write_json(directory/'state.json', state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'case', 'pilots', 'production'])
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(), flush=True)
        return
    if args.campaign is None:
        parser.error('--campaign is required')
    directory = args.campaign.resolve()
    if args.action == 'case':
        if args.name not in json.loads((directory/'request.json').read_text())['cases']:
            parser.error('Valid --name is required')
        result = execute_case(directory, args.name)
        print(json.dumps(result), flush=True)
        if not result['diagnostic_qualification']:
            raise SystemExit(1)
    else:
        if args.action == 'production':
            checks = json.loads((directory/'pilot_checks.json').read_text())
            if checks.get('status') != 'passed_startup_only':
                raise ValueError('Production requires an explicit passed startup pilot report')
        names = ['re10000', 're30000'] if args.action == 'production' else [
            'pilot_direct', 'pilot_recycled', 'pilot_high', 'pilot_time', 'pilot_mesh']
        drive(directory, names, render=args.action == 'production')


if __name__ == '__main__':
    main()

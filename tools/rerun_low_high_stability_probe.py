"""Rerun the saved 64^2 low/high MMS with the current imported solver.

Controls: reuse physical parameters, fixed step, initial state and norm cutoff.
Observe every node's exact semidiscrete error and norm, plus actual frozen-factor
stage corrections. Completion/boundedness alone does not establish accuracy.
This entry point computes and saves; plotting is a separate saved-data operation.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.records import begin_run, digest, provenance, write_json
from solver.core import Scheme, Stage
from solver.mac.grid import Array, MACGrid
from solver.mac_ns import MACNavierStokes
from solver.schemes.sdirk3 import SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV

BASELINE = '20261004T080538558295'
HISTORY_COLUMNS = ['t', 'relative_L2_error', 'norm_over_initial', 'r', 'max_stage_abs_r',
                   'max_stage_abs_coupling_minus_1', 'divergence_inf',
                   'momentum_residual', 'scalar_residual']
STAGE_COLUMNS = ['r', 'predictor', 'correction_1', 'correction_2', 'frozen_factor',
                 'coupling', 'momentum_residual', 'scalar_residual', 'divergence_inf',
                 'continuity_residual', 'correction_count']


def baseline_config() -> dict[str, Any]:
    import json
    old = json.loads((ROOT/'runs/low_high_stability_probe'/BASELINE/'metadata.json').read_text())
    cfg = old['configuration'].copy()
    cfg['dt'], cfg['gamma'] = cfg['dt'][0], cfg['gamma'][0]
    cfg['r0'], cfg['norm_ratio_cutoff'] = 0., 100.
    return cfg


def mode(grid: MACGrid, q: int) -> Array:
    x, y = grid.coordinates('vertex')
    envelope = np.sin(np.pi*x)**2*np.sin(np.pi*y)**2
    psi = envelope if q == 0 else envelope*np.sin(2*np.pi*q*x)*np.sin(2*np.pi*q*y)
    psi[[0, -1], :] = 0.
    psi[:, [0, -1]] = 0.
    u, v = np.diff(psi, axis=0)/grid.hy, -np.diff(psi, axis=1)/grid.hx
    vec = grid.pack(u, v)
    return vec/grid.norm(vec)


def problem(cfg: dict[str, Any]) -> tuple[MACNavierStokes, Callable[[float], Array], dict[str, float]]:
    if cfg['ramp'] != 0.:
        raise ValueError('This rerun preserves the constant low mode; ramp must be zero')
    grid = MACGrid(cfg['nx'], cfg['nx'])
    model = MACNavierStokes(grid, cfg['nu'])
    low, high = mode(grid, 0), mode(grid, cfg['q'])
    kl, kh = model.apply_K(low), model.apply_K(high)
    nl, nh = model.nonlinear(low), model.nonlinear(high)
    cross = model.nonlinear(low+high)-nl-nh

    def exact(t: float) -> Array:
        return cfg['low']*low+cfg['epsilon']*np.exp(-cfg['beta']*t)*high

    def force(t: float) -> Array:
        a, b = cfg['low'], cfg['epsilon']*np.exp(-cfg['beta']*t)
        return (0.*low-cfg['beta']*b*high+cfg['nu']*(a*kl+b*kh)
                +a*a*nl+a*b*cross+b*b*nh)

    model.force = force
    checks = {'mode_inner': grid.inner(low, high)}
    for t in (0., .71, cfg['T']):
        vel = exact(t)
        derivative = -cfg['beta']*cfg['epsilon']*np.exp(-cfg['beta']*t)*high
        defect = derivative+model.nu*model.apply_K(vel)+model.nonlinear(vel)-force(t)
        checks[f'residual_{t}'] = model.max_abs(defect)
        checks[f'divergence_{t}'] = model.physical_divergence_inf(vel)
        checks[f'convection_work_{t}'] = model.inner(model.nonlinear(vel), vel)
        assert checks[f'residual_{t}'] < 1e-10
        assert checks[f'divergence_{t}'] < 1e-10
        assert abs(checks[f'convection_work_{t}']) < 1e-10
    return model, exact, checks


def stage_row(stage: Stage, sav: bool) -> list[float]:
    if sav:
        assert len(stage.corrections) == 2 and not stage.candidates
        first, last = stage.corrections
        assert last == stage.r
        np.testing.assert_allclose(stage.frozen_factor, 1+first+first*first, rtol=1e-14)
        return [stage.r, stage.correction_predictor, first, last, stage.frozen_factor,
                stage.frozen_factor*(1-stage.r), stage.residual, stage.scalar_residual,
                stage.divergence_inf, stage.continuity_residual, 2.]
    return [stage.r, np.nan, np.nan, np.nan, np.nan, 1., stage.residual,
            stage.scalar_residual, stage.divergence_inf, stage.continuity_residual, 0.]


def run_case(cfg: dict[str, Any], scheme: Scheme, out: Path, logger: logging.Logger) -> dict[str, Any]:
    model, exact, checks = problem(cfg)
    state = model.state(0., exact(0.), cfg['r0'])
    initial_norm = model.grid.norm(model.vector(state))
    hist: list[list[float]] = []
    stage_history: list[list[list[float]]] = []
    sav = scheme.name == 'sdirk3_mrsav'
    gamma = getattr(scheme, 'gamma', None)
    member = f'{scheme.name}-dt{cfg["dt"]}-g{gamma if sav else 0}'
    info = {'config': cfg, 'scheme': scheme.name, 'gamma': gamma, 'status': 'running',
            'checks': checks, 'schema_version': 2,
            'scalar_solver': 'fixed_count_frozen_factor' if sav else 'none'}
    write_json(out/f'{member}.json', info)

    def record(stages: list[Stage] | None = None) -> None:
        vec, truth = model.vector(state), exact(state.t)
        rows = [stage_row(stage, sav) for stage in stages] if stages else []
        if rows:
            stage_history.append(rows)
        hist.append([state.t, model.grid.norm(vec-truth)/model.grid.norm(truth),
                     model.grid.norm(vec)/initial_norm, state.r,
                     max(abs(row[0]) for row in rows) if rows else abs(state.r),
                     max(abs(row[5]-1) for row in rows) if rows else 0.,
                     model.physical_divergence_inf(vec),
                     max(stage.residual for stage in stages) if stages else 0.,
                     max(stage.scalar_residual for stage in stages) if stages else 0.])

    record()
    start, status, error = perf_counter(), 'running', None
    try:
        count = int(np.ceil(cfg['T']/cfg['dt']-1e-12))
        for i in range(count):
            end = min((i+1)*cfg['dt'], cfg['T'])
            step = cfg['dt'] if i < count-1 else end-state.t
            trial = scheme.step(model, state, step)
            state = replace(trial.state, t=end)
            record(trial.stages)
            if (i+1) % cfg['log_every'] == 0:
                logger.info('%s step=%d t=%.8g relative_L2_error=%.9g elapsed=%.2fs',
                            scheme.name, i+1, state.t, hist[-1][1], perf_counter()-start)
            if not np.isfinite(hist[-1]).all():
                status = 'nonfinite'
                break
            if hist[-1][2] > cfg['norm_ratio_cutoff']:
                status = 'solution_blowup_norm_ratio_gt_100'
                break
        else:
            status = 'complete'
    except Exception as exc:
        status, error = 'solver_failed', repr(exc)
        logger.exception('%s stopped', scheme.name)
    arr = np.asarray(hist)
    pending = out/f'{member}.pending.npz'
    np.savez_compressed(pending, history=arr, final_velocity=model.vector(state),
                        stage_times=arr[1:, 0], stage_records=np.asarray(stage_history).reshape(-1, 4, 11))
    pending.replace(out/f'{member}.npz')
    summary = {**info, 'status': status, 'exception': error, 'seconds': perf_counter()-start,
               'accepted_steps': len(arr)-1, 't_final': state.t,
               'final_rel_l2': float(arr[-1, 1]), 'max_rel_l2': float(arr[:, 1].max()),
               'early_max_rel_l2_t_le_5': float(arr[arr[:, 0] <= 5., 1].max()),
               'max_norm_ratio': float(arr[:, 2].max()), 'min_norm_ratio': float(arr[:, 2].min()),
               'max_stage_r': float(arr[:, 4].max()),
               'max_stage_abs_coupling_minus_1': float(arr[:, 5].max()),
               'max_divergence': float(arr[:, 6].max()),
               'max_momentum_residual': float(arr[:, 7].max()),
               'max_scalar_residual': float(arr[:, 8].max())}
    write_json(out/f'{member}.json', summary)
    logger.info('CASE_FINAL %s', summary)
    return summary


def run(cfg: dict[str, Any], parent: Path) -> Path:
    if not np.isfinite([cfg[k] for k in ('T', 'dt', 'nu', 'gamma', 'r0')]).all() \
            or min(cfg['T'], cfg['dt'], cfg['nu']) <= 0 or cfg['gamma'] < 0:
        raise ValueError('Invalid effective configuration')
    source = provenance()
    out, manifest = begin_run(parent, cfg, source,
                              metadata={'experiment': 'low_high_stability_probe', 'schema_version': 2})
    logger = logging.getLogger(out.name)
    logger.setLevel(logging.INFO)
    handlers: list[logging.Handler] = [logging.FileHandler(out/'run.log'), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    try:
        files = [Path(__file__), ROOT/'experiments/records.py', *sorted((ROOT/'solver').rglob('*.py')),
                 ROOT/'pyproject.toml', ROOT/'uv.lock', ROOT/'.python-version']
        hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
        for relative in hashes:
            saved = out/'source'/relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, saved)
        write_json(out/'metadata.json', {'configuration': cfg, 'source': source,
                   'source_sha256': hashes, 'baseline_run_id': BASELINE,
                   'history_columns': HISTORY_COLUMNS, 'stage_columns': STAGE_COLUMNS,
                   'stage_records_alignment': 'stage_times=history[1:,0]',
                   'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
        logger.info('START run=%s config=%s', out, cfg)
        summaries = [run_case(cfg, scheme, out, logger) for scheme in (SDIRK3(), SDIRK3MRSAV(cfg['gamma']))]
        write_json(out/'summary.json', {'members': summaries})
        unchanged = all(digest(ROOT/relative) == checksum for relative, checksum in hashes.items())
        assert unchanged, 'Numerical sources changed during the run'
        names = ['metadata.json', 'config.json', 'summary.json']
        names += [path.name for path in out.glob('sdirk3*') if path.suffix in ('.json', '.npz')]
        archive = {'run_id': out.name, 'schema_version': 2,
                   'members': [path.stem for path in sorted(out.glob('sdirk3*.npz'))],
                   'sha256': {name: digest(out/name) for name in names}, 'source_unchanged': unchanged}
        write_json(out/'archive_manifest.json', archive)
        manifest.update(status='finished', members=summaries, source_unchanged=unchanged,
                        archive_manifest_sha256=digest(out/'archive_manifest.json'))
        write_json(out/'manifest.json', manifest)
        logger.info('FINAL run=%s statuses=%s', out, [member['status'] for member in summaries])
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'runs/low_high_stability_probe')
    parser.add_argument('--nx', type=int, help='Smoke-test override; production reuses the baseline 64')
    parser.add_argument('--T', type=float, help='Smoke-test override; production reuses the baseline 30')
    args = parser.parse_args()
    cfg = baseline_config()
    for key in ('nx', 'T'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    print(run(cfg, args.output_root))


if __name__ == '__main__':
    main()

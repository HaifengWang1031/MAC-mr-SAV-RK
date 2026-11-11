"""Explicit-evidence registry: what each stored run is allowed to support.

Read-only. Scans the explicit runs root, verifies each record through its reader and
writes a table that keeps four states apart:

``complete``      the computation finished and the record validates;
``qualification`` the run passes the numerical gates recorded in its own summary;
``statistics``    the finite-window stationarity screen passed;
``excluded``      the record must not be used as physical evidence.

Nothing here launches or resumes a computation, and no run is ever rewritten.
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import h5py

from experiments.records import PROJECT, write_json
from .analyze import block_statistics
from .records import load

COUPLING_LIMIT = 1e-3
EXCLUSION_REASONS = {
    'not_complete': 'manifest status is not complete; no published results',
    'small_gamma': 'measured max|G-1| above the coupling limit: the SAV factor suppressed the nonlinear term',
    'pilot_resolution': 'reduced-Re/kfL pilot (below the paper grid): pipeline evidence only, no paper comparison',
    'duplicate': 'duplicate of another record with the same configuration',
}


def _scan(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            c, manifest = load(directory)
        except Exception as error:  # unreadable or unfinished attempts are evidence too
            rows.append({'run': directory.name, 'status': 'unreadable',
                         'exclusion': 'not_complete', 'reason': f'{type(error).__name__}: {error}'})
            continue
        rows.append({'run': directory.name, 'status': manifest['status'],
                     'config': {key: c[key] for key in
                                ('nx', 'ny', 'kf', 'mu', 'Re_plus', 'Re_minus', 'gamma', 'dt',
                                 'T', 'sample_every', 'seed', 'scheme')},
                     'identity': manifest['identity'], 'results_sha256': manifest['results_sha256'],
                     'metrics': manifest.get('metrics', {})})
    return rows


def _mark(rows: list[dict[str, Any]], paper_kf: int = 8, paper_re_plus: float = 1400.) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        if row['status'] != 'complete':
            row.update(exclusion='not_complete', reason=EXCLUSION_REASONS['not_complete'])
            continue
        deviation = row.get('max_abs_coupling_minus_one')
        if deviation is None or not np.isfinite(deviation):
            row.update(exclusion='small_gamma',
                       reason='coupling deviation could not be measured from this record')
            continue
        if deviation > COUPLING_LIMIT:
            row.update(exclusion='small_gamma', reason=EXCLUSION_REASONS['small_gamma'])
            continue
        if row['config']['kf'] < paper_kf or row['config']['Re_plus'] < paper_re_plus:
            row.update(exclusion='pilot_resolution', reason=EXCLUSION_REASONS['pilot_resolution'])
            continue
        key = json.dumps(row['config'], sort_keys=True)
        if key in seen:
            row.update(exclusion='duplicate', reason=f'duplicate of {seen[key]}')
            continue
        seen[key] = row['run']
        row.update(exclusion=None, reason='')


def _statistics(row: dict[str, Any], root: Path, burn_in: float, block: float) -> None:
    """Attach the finite-window screen; a missing screen is reported, not guessed."""
    if row.get('exclusion'):
        return
    try:
        c, _ = load(root/row['run'], require_complete=True)
        with h5py.File(root/row['run']/'results.h5', 'r') as d:
            ts = d['spectra/times'][:]/c['t0']
            stats = block_statistics(ts, d['spectra/Pi_E'][:, int(np.floor(.75*np.sqrt(2)*c['kf']))],
                                     start=burn_in, duration=block)
            coupling = (d['corrections/frozen_factor'][:]*(1-d['stages/r'][:])
                        if c['scheme'].endswith('mrsav') else np.ones_like(d['stages/r'][:]))
            row['max_abs_coupling_minus_one'] = float(np.max(np.abs(coupling-1)))
        row['statistics'] = {'mean_Pi_E': stats['mean'], 'std_Pi_E': stats['std'],
                             'n_blocks': stats['n_blocks'],
                             'tau_int_Pi_E': stats['correlation'].get('tau_int'),
                             'n_eff_Pi_E': stats['correlation'].get('n_eff'),
                             'tail_dropped_t0': stats['tail_dropped_t0']}
    except Exception as error:
        row['statistics'] = {'status': 'unavailable', 'reason': f'{type(error).__name__}: {error}'}


def _fmt(value: Any) -> str:
    return '' if value is None else f'{value:.1e}'


def report(rows: list[dict[str, Any]], burn_in: float, block: float) -> str:
    lines = ['# Evidence registry (generated)', '',
             f'coupling limit max|G-1| = {COUPLING_LIMIT:g}; burn-in t*={burn_in:g}; '
             f'block t*={block:g}.',
             'Only records with an empty `excluded` cell may support physical statements.', '',
             '| run | status | N | mu | Re+ | Re- | gamma | T* | seed | max|G-1| | mean Pi_E | excluded |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        if 'config' in row:
            c = row['config']
            mean = (row.get('statistics') or {}).get('mean_Pi_E')
            lines.append(f"| `{row['run']}` | {row['status']} | {c['nx']} | {c['mu']:g} | "
                         f"{c['Re_plus']:g} | {c['Re_minus']:g} | {c['gamma']:g} | {c['T']:g} | "
                         f"{c['seed']} | {_fmt(row.get('max_abs_coupling_minus_one'))} | "
                         f"{'' if mean is None else f'{mean:+.5f}'} | {row.get('exclusion') or ''} |")
        else:
            lines.append(f"| `{row['run']}` | {row['status']} | - | - | - | - | - | - | - | - | - | "
                         f"{row.get('exclusion') or ''} |")
    usable = [row for row in rows if row.get('exclusion') is None]
    lines += ['', f'Usable records: {len(usable)} of {len(rows)}.']
    return '\n'.join(lines)+'\n'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=PROJECT/'runs'/'inverse_cascade')
    p.add_argument('--output', type=Path, default=PROJECT/'reports'/'inverse_cascade'/'inventory')
    p.add_argument('--burn-in', type=float, default=40.)
    p.add_argument('--block', type=float, default=5.)
    a = p.parse_args()
    rows = _scan(a.root)
    for row in rows:
        _statistics(row, a.root, a.burn_in, a.block)
    _mark(rows)
    a.output.mkdir(parents=True, exist_ok=True)
    write_json(a.output/'inventory.json',
               {'coupling_limit': COUPLING_LIMIT, 'burn_in_t0': a.burn_in,
                'block_t0': a.block, 'runs': rows})
    (a.output/'inventory.md').write_text(report(rows, a.burn_in, a.block))
    print(a.output/'inventory.md')


if __name__ == '__main__':
    main()

"""Pre-registered experiment matrix and resource budget; dry run by default.

This module never contacts a cluster and never submits work as a side effect. It
prints and writes the matrix (configs, purpose, acceptance gates, cost estimates) so
that expensive stages can be approved one at a time. ``--execute --only ID`` runs the
named entry in this process through the segmented path; nothing is spawned remotely.

Cost fields are estimates. ``seconds_per_step`` uses measured values where available
and a documented N^2 log N extrapolation beyond 512^2; replace them with a benchmark
before approving any stage.
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.records import PROJECT, digest, provenance, write_json
from .config import effective_config

# Measured on this machine (SDIRK3-mr-ccSAV, one process, FFT workers=1).
MEASURED_SECONDS_PER_STEP = {96: 1.1e-2, 192: 4.1e-2, 256: 7.3e-2, 512: 3.2e-1}
BYTES_PER_STEP = 872.  # diagnostics/stages/corrections for one accepted step at 512^2
SPECTRA_COLUMNS = 12   # four flux/energy columns plus eight low-pass budget columns
BUDGET_MARGIN = 1.30
PLAN_SCHEMA_VERSION = 2
PAPER_SCALES = {8: (512, 2.3e4), 16: (1024, 7.4e5),
                32: (2048, 2.3e7), 64: (4096, 7.6e8)}

BASE = {'kf': 8, 'f0': 1., 'Re_plus': 1400., 'Re_minus': 23000.,
        'gamma': 1000., 'dt': .005, 'sample_every': 20, 'seed': 2014,
        'noise_rms': .01, 'coupling_tolerance': 1e-3}

PAPER = {'kfL': (8, 16, 32, 64), 'Re_minus': (2.3e4, 7.4e5, 2.3e7, 7.6e8),
         'Re_plus': 1400., 'mu_c_energy': .22, 'mu_c_potential': .25}


def _scaling(n: int) -> float:
    """Measured seconds per step, or an N^2 log N extrapolation above 512^2."""
    if n in MEASURED_SECONDS_PER_STEP:
        return MEASURED_SECONDS_PER_STEP[n]
    base = max(MEASURED_SECONDS_PER_STEP)
    return MEASURED_SECONDS_PER_STEP[base]*(n/base)**2*(np.log2(n)/np.log2(base))


def shells(n: int) -> int:
    """Radial shell count for the strict 2/3 retained Fourier band."""
    return int(np.ceil(np.sqrt(2)*(np.ceil(n/3)-1)))+1


def budget(n: int, T: float, dt: float, sample_every: int) -> dict[str, Any]:
    steps = int(round(T/dt))
    per_step = _scaling(n)
    samples = steps//sample_every+1
    storage_history = steps*BYTES_PER_STEP
    storage_spectra = samples*shells(n)*SPECTRA_COLUMNS*8
    memory = n*n*32*20  # ~20 copies of one (2,2,ny,nx/2+1) complex state, rough
    return {'grid': f'{n}x{n}', 'steps': steps, 'seconds_per_step': per_step,
            'wall_hours': steps*per_step/3600, 'peak_memory_gb': memory/2**30,
            'storage_gb': (storage_history+storage_spectra)/2**30,
            'storage_spectra_gb': storage_spectra/2**30,
            'wall_hours_with_margin': BUDGET_MARGIN*steps*per_step/3600,
            'storage_gb_with_margin': BUDGET_MARGIN*(storage_history+storage_spectra)/2**30,
            'budget_margin': BUDGET_MARGIN,
            'cost_basis': ('historical_local_measurement' if n in MEASURED_SECONDS_PER_STEP
                           else 'extrapolated'),
            'limits': ('Not a host benchmark: historical timing predates low-pass columns; '
                       'memory is a working-array estimate only, excludes Python/FFT/HDF5 '
                       'overhead and per-segment full-state checkpoints. No measured peak RSS.')}


def config_digest(config: dict[str, Any]) -> str:
    """Canonical checksum of the *effective* configuration, not an entry's name."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_matrix(entries: list[dict[str, Any]]) -> None:
    """Reject stale names, unresolved/forward dependencies and mismatched identities."""
    seen: set[str] = set()
    for item in entries:
        if item['id'] in seen or not set(item['depends_on']) <= seen:
            raise ValueError('Duplicate entry or missing/forward dependency')
        c = effective_config(item['config'])
        if config_digest(c) != item['config_sha256']:
            raise ValueError('Effective configuration checksum mismatch')
        if item['stage'] == 'scale':
            kf = int(item['id'].split('_')[0][2:])
            n, re_minus = PAPER_SCALES[kf]
            if (c['kf'], c['nx'], c['ny'], c['Re_minus'], c['Re_plus']) != \
                    (kf, n, n, re_minus, 1400.):
                raise ValueError('Scale entry does not implement its paper parameters')
        seen.add(item['id'])


def _evidence(items: list[dict[str, str]], base: Path) -> None:
    if not items:
        raise ValueError('Approval requires explicit evidence files')
    for evidence in items:
        path = base/evidence['path']
        if not path.is_file() or digest(path) != evidence['sha256']:
            raise ValueError(f'Approval evidence checksum mismatch: {path}')


def validate_approval(item: dict[str, Any], path: Path, entries: list[dict[str, Any]]) -> None:
    """Fail closed before computation; human review is not automated physical proof.

    An approval binds the effective config and current code to a named reviewer,
    resource evidence and *every* prerequisite's reviewed, hashed evidence. There
    is no generated pre-approved template and no implicit latest-report search.
    """
    approval = json.loads(path.read_text())
    if approval.get('approval_schema_version') != 1 or not approval.get('reviewed_by', '').strip():
        raise ValueError('An explicit named review is required')
    if approval.get('entry_id') != item['id'] or approval.get('config_sha256') != item['config_sha256']:
        raise ValueError('Approval entry/config identity mismatch')
    if approval.get('code_sha256') != provenance()['code_sha256']:
        raise ValueError('Approval was not reviewed for the current code')
    resource = approval.get('resource_review', {})
    if resource.get('passed') is not True:
        raise ValueError('Resource approval is required before execution')
    _evidence(resource.get('evidence', []), path.parent)
    # Every high-resolution scale needs its own qualification, not just the 512^2 predecessor.
    if item['stage'] == 'scale':
        local = approval.get('scale_qualification', {})
        if local.get('config_sha256') != item['config_sha256'] or local.get('passed') is not True:
            raise ValueError('Own scale grid/step qualification is required')
        _evidence(local.get('evidence', []), path.parent)
    by_id = {entry['id']: entry for entry in entries}
    reviews = approval.get('dependencies', [])
    if len(reviews) != len(item['depends_on']) or \
            {r['entry_id'] for r in reviews} != set(item['depends_on']):
        raise ValueError('Approval must review every declared dependency exactly once')
    for review in reviews:
        dependency = by_id[review['entry_id']]
        required = {'numerical'} if dependency['stage'] == 'qualification' else {'numerical', 'statistics'}
        checks = review.get('checks', {})
        if review.get('config_sha256') != dependency['config_sha256'] or \
                any(checks.get(key) is not True for key in required):
            raise ValueError('Dependency qualification missing or configuration mismatch')
        _evidence(review.get('evidence', []), path.parent)


def entry(entry_id: str, purpose: str, stage: str, config: dict[str, Any],
          gate: str, depends_on: tuple[str, ...] = ()) -> dict[str, Any]:
    c = effective_config({**BASE, **config})
    cost = budget(c['nx'], c['T'], c['dt'], c['sample_every'])
    return {'id': entry_id, 'stage': stage, 'purpose': purpose, 'config': {**BASE, **config},
            'config_sha256': config_digest(c),
            'gate': gate, 'depends_on': list(depends_on), **cost}


def matrix() -> list[dict[str, Any]]:
    """The pre-registered stages in dependency order; nothing here is executed."""
    entries: list[dict[str, Any]] = []
    # Stage 5: numerical qualification, cheap enough to run on one machine.
    for n, dt in ((192, .005), (256, .005), (256, .0025), (512, .005)):
        entries.append(entry(f'qual_{n}_{str(dt).replace(".", "")}', 'grid/step qualification',
                             'qualification', {'nx': n, 'ny': n, 'mu': .2, 'T': 100., 'dt': dt},
                             'low-pass window defect <= 1e-3, coupling <= 1e-3, '
                             'flux mean within combined stderr and 5% of injection'))
    # Stage 6: credible steady state at the paper's smallest scale.
    for mu in (.10, .20, .25, .30, .80):
        entries.append(entry(f'steady512_mu{int(mu*100):03d}', 'steady-state window at kfL=8',
                             'steady', {'nx': 512, 'ny': 512, 'mu': mu, 'T': 500.},
                             'stationarity screen passes in two consecutive windows',
                             ('qual_512_0005',)))
    for mu in (.20, .25, .80):
        for seed in (7, 99):
            entries.append(entry(f'steady512_mu{int(mu*100):03d}_s{seed}', 'seed spread',
                                 'steady', {'nx': 512, 'ny': 512, 'mu': mu, 'T': 500., 'seed': seed},
                                 'seed spread below block halfwidth', ('qual_512_0005',)))
    # Stage 7: critical region, only after the steady gate passes.
    for mu in (0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30):
        entries.append(entry(f'crit512_mu{round(mu*100):03d}', 'critical-region refinement',
                             'critical', {'nx': 512, 'ny': 512, 'mu': mu, 'T': 500.},
                             'sign change supported by three cutoffs and the dissipation split',
                             ('steady512_mu020',)))
    # Stage 8: paper Fig. 1-4 scale hierarchy; each needs its own qualification.
    for kfL, n in ((16, 1024), (32, 2048), (64, 4096)):
        for mu in (.20, .22, .26):
            entries.append(entry(f'kf{kfL}_{n}_mu{round(mu*100):03d}', f'Fig.1/Fig.2 at kfL={kfL}',
                                 'scale', {'nx': n, 'ny': n, 'kf': kfL,
                                           'Re_minus': PAPER_SCALES[kfL][1], 'mu': mu, 'T': 300.},
                                 'own grid/step qualification; no inherited dt or burn-in',
                                 ('crit512_mu022',)))
    return entries


def report(entries: list[dict[str, Any]]) -> str:
    lines = ['# Pre-registered matrix (dry run, nothing submitted)', '',
             f'{len(entries)} entries. Cost basis: measured seconds/step up to 512^2, '
             'N^2 log N extrapolation above; replace by a benchmark before approval.',
             'This is a partial candidate matrix, NOT full Fig.1–4 coverage. Each scale still needs '
             'its own grid/step and resource review; dt is a candidate, never inherited qualification.',
             'All costs are host-unverified estimates; approval uses 30% margin. Checkpoint storage '
             'and peak RSS must be measured separately.', '']
    lines.append('| id | stage | grid | T* | wall h | peak GB | storage GB | gate |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for item in entries:
        lines.append(f"| {item['id']} | {item['stage']} | {item['grid']} | {item['config']['T']:g} | "
                     f"{item['wall_hours']:.1f} | {item['peak_memory_gb']:.1f} | "
                     f"{item['storage_gb']:.2f} | {item['gate']} |")
    total = sum(item['wall_hours'] for item in entries)
    storage = sum(item['storage_gb'] for item in entries)
    lines += ['', f'Total wall hours (serial): {total:.0f}', f'Total storage: {storage:.1f} GB', '']
    by_stage: dict[str, float] = {}
    for item in entries:
        by_stage[item['stage']] = by_stage.get(item['stage'], 0.)+item['wall_hours']
    lines += ['| stage | wall h |', '| --- | --- |']
    lines += [f'| {stage} | {hours:.0f} |' for stage, hours in by_stage.items()]
    return '\n'.join(lines)+'\n'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=PROJECT/'reports'/'inverse_cascade'/'plan')
    p.add_argument('--execute', action='store_true', help='run the selected entry locally')
    p.add_argument('--only', type=str, default=None, help='entry id to run with --execute')
    p.add_argument('--write-configs', action='store_true',
                   help='emit one JSON config per entry under <output>/configs')
    p.add_argument('--approval', type=Path, help='explicit reviewed approval with hashed evidence')
    a = p.parse_args()
    entries = matrix()
    validate_matrix(entries)
    selected = None
    if a.execute:
        if not a.only or not a.approval:
            p.error('--execute requires --only ENTRY_ID and --approval FILE')
        selected = next((item for item in entries if item['id'] == a.only), None)
        if selected is None:
            p.error(f'unknown entry: {a.only}')
        validate_approval(selected, a.approval, entries)
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Use a new output directory; historical plans are not overwritten')
    a.output.mkdir(parents=True, exist_ok=True)
    write_json(a.output/'matrix.json', {'plan_schema_version': PLAN_SCHEMA_VERSION,
                                       'paper': PAPER, 'base': BASE, 'entries': entries})
    (a.output/'matrix.md').write_text(report(entries))
    if a.write_configs:
        (a.output/'configs').mkdir(parents=True, exist_ok=True)
        for item in entries:
            write_json(a.output/'configs'/f"{item['id']}.json", {**BASE, **item['config']})
    if not a.execute:
        print(report(entries))
        print(f'written: {a.output}/matrix.json, {a.output}/matrix.md')
        return
    assert selected is not None
    from .segments import run_segmented
    item = selected
    print(f'running {item["id"]} locally (segmented): {item["config"]}', flush=True)
    path = run_segmented({**BASE, **item['config']}, execution={'segment_steps': 1000})
    print(path)


if __name__ == '__main__':
    main()

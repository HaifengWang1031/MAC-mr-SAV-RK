"""Search fixed steps at unchanged MMS parameters, retaining every outcome.

Qualify only late ordinary norm blowup and complete SAV integration with a small
maximum-in-time exact relative error. Rank qualifying steps by that maximum,
not final error. This computes runs and a search index; analysis is separate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.records import begin_run, digest, provenance, write_json
from tools.rerun_low_high_stability_probe import baseline_config, run

DEFAULT_STEPS = [.00387, .0038725, .0038775, .003885, .0039, .004]


def worker(cfg: dict[str, Any], batch: Path) -> Path:
    with (batch/f'worker-dt{cfg["dt"]}.log').open('w') as stream, \
            redirect_stdout(stream), redirect_stderr(stream):
        return run(cfg, batch/'cases')


def summarize(folder: Path, batch: Path, error_limit: float) -> dict[str, Any]:
    members = json.loads((folder/'summary.json').read_text())['members']
    ordinary, sav = members
    cfg = ordinary['config']
    qualifies = (ordinary['status'] == 'solution_blowup_norm_ratio_gt_100'
                 and ordinary['t_final'] > 5. and sav['status'] == 'complete'
                 and sav['t_final'] == cfg['T'] and sav['max_rel_l2'] < error_limit
                 and max(m['early_max_rel_l2_t_le_5'] for m in members) < 1e-5)
    return {'dt': cfg['dt'], 'run': str(folder.relative_to(batch)), 'qualifies': qualifies,
            'ordinary_status': ordinary['status'], 'ordinary_t_final': ordinary['t_final'],
            'ordinary_max_norm_ratio': ordinary['max_norm_ratio'],
            'ordinary_early_max_error': ordinary['early_max_rel_l2_t_le_5'],
            'sav_status': sav['status'], 'sav_t_final': sav['t_final'],
            'sav_max_error': sav['max_rel_l2'], 'sav_final_error': sav['final_rel_l2'],
            'sav_early_max_error': sav['early_max_rel_l2_t_le_5'],
            'sav_max_norm_ratio': sav['max_norm_ratio'],
            'archive_manifest_sha256': digest(folder/'archive_manifest.json')}


def search(cfg: dict[str, Any], steps: list[float], parent: Path, *,
           workers: int = 3, error_limit: float = .01) -> Path:
    if not steps or len(set(steps)) != len(steps) or not np.isfinite(steps).all() \
            or min(steps) <= 0 or workers < 1 or not np.isfinite(error_limit) or error_limit <= 0:
        raise ValueError('Require distinct positive finite steps, workers and error limit')
    controls = {'base': cfg, 'steps': steps, 'workers': workers, 'error_limit': error_limit,
                'early_time': 5., 'early_error_limit': 1e-5,
                'rank_by': 'SAV maximum relative L2 error over every saved node'}
    batch, manifest = begin_run(parent, controls, provenance(),
                               metadata={'experiment': 'low_high_step_search', 'schema_version': 1})
    files = [Path(__file__), ROOT/'tools/rerun_low_high_stability_probe.py',
             ROOT/'experiments/records.py', *sorted((ROOT/'solver').rglob('*.py')),
             ROOT/'pyproject.toml', ROOT/'uv.lock', ROOT/'.python-version']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    for relative in hashes:
        target = batch/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    write_json(batch/'source_manifest.json', hashes)
    logger = logging.getLogger(batch.name)
    logger.setLevel(logging.INFO)
    handlers: list[logging.Handler] = [logging.FileHandler(batch/'search.log'), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    rows: list[dict[str, Any]] = []
    try:
        logger.info('START batch=%s steps=%s workers=%d', batch, steps, workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(worker, {**cfg, 'dt': dt}, batch): dt for dt in steps}
            for future in as_completed(pending):
                dt = pending[future]
                try:
                    row = summarize(future.result(), batch, error_limit)
                    logger.info('CASE_FINAL dt=%.8g ordinary=%s t=%.8g SAV=%s max_error=%.9g final_error=%.9g qualifies=%s',
                                dt, row['ordinary_status'], row['ordinary_t_final'], row['sav_status'],
                                row['sav_max_error'], row['sav_final_error'], row['qualifies'])
                except Exception as exc:
                    row = {'dt': dt, 'qualifies': False, 'error': repr(exc)}
                    logger.exception('CASE_ERROR dt=%.8g', dt)
                rows.append(row)
                rows.sort(key=lambda item: item['dt'])
                write_json(batch/'search.json', {'status': 'running', 'configuration': controls, 'cases': rows})
        assert all(digest(ROOT/name) == value for name, value in hashes.items()), 'Search sources changed'
        qualified = [row for row in rows if row['qualifies']]
        best = min(qualified, key=lambda row: row['sav_max_error']) if qualified else None
        record = {'status': 'complete', 'configuration': controls, 'cases': rows,
                  'best': best, 'source_unchanged': True,
                  'failed_case_count': sum('error' in row for row in rows)}
        write_json(batch/'search.json', record)
        manifest.update(status='complete', search_sha256=digest(batch/'search.json'),
                        completed_cases=len(rows), qualified_cases=len(qualified),
                        failed_cases=record['failed_case_count'])
        write_json(batch/'manifest.json', manifest)
        logger.info('FINAL batch=%s qualified=%d/%d best=%s', batch, len(qualified), len(rows), best)
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=float, nargs='+', default=DEFAULT_STEPS)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--error-limit', type=float, default=.01)
    parser.add_argument('--nx', type=int)
    parser.add_argument('--T', type=float)
    parser.add_argument('--output-root', type=Path,
                        default=ROOT/'runs/low_high_stability_probe/step_search')
    args = parser.parse_args()
    cfg = baseline_config()
    for key in ('nx', 'T'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    print(search(cfg, args.steps, args.output_root, workers=args.workers, error_limit=args.error_limit))


if __name__ == '__main__':
    main()

"""Read explicitly selected historical reference-error reports, without computation."""
import csv
import json
import math
from pathlib import Path

import h5py

from .workflow import digest, load_record


def read_reference_study(root: Path, report: Path, *, analysis_sha256: str,
                         table_sha256: str) -> tuple[dict, dict, list[dict[str, str]]]:
    """Return the captured batch, recorded physical config and saved error rows.

    This is presentation of pinned historical reports, not reuse under today's
    source identity or independent recomputation of field errors.
    """
    root, report = Path(root).resolve(), Path(report).resolve()
    table = report/'tables/errors.csv'
    if digest(report/'analysis.json') != analysis_sha256 or digest(table) != table_sha256:
        raise ValueError('Presentation input checksum mismatch')
    analysis = json.loads((report/'analysis.json').read_text())
    if analysis['status'] != 'complete' or analysis.get('reference_passed') is not True:
        raise ValueError('A completed report with verified reference sensitivity is required')
    if digest(report/'input_batch.json') != analysis['batch_sha256']:
        raise ValueError('Captured batch checksum mismatch')
    batch = json.loads((report/'input_batch.json').read_text())
    if batch['source'] != analysis['source']:
        raise ValueError('Captured batch source mismatch')
    base = batch['config']['base']
    experiment = base['experiment']
    if experiment not in ('forced_ns', 'trig_ns'):
        raise ValueError('Not a supported reference-error study')
    times = [float(t) for t in base['snapshots']]
    checks = analysis.get('reference_checks', [])
    threshold = batch['config']['sensitivity_threshold']
    if ({(c['time'], c['norm']) for c in checks} != {(t, n) for t in times for n in ('L2', 'H1')}
            or any(c['passed'] is not True or not math.isfinite(c['gap_over_min_error'])
                   or not 0 <= c['gap_over_min_error'] <= threshold for c in checks)):
        raise ValueError('Reference sensitivity checks are incomplete or not passed')
    records: dict[str, tuple[dict, dict]] = {}
    for item in analysis['inputs']:
        saved = Path(item['path'])
        if saved.parent.name != experiment or saved.name in records:
            raise ValueError('Ambiguous historical input identity')
        directory = root/'runs'/experiment/saved.name
        cfg, manifest = load_record(directory)
        if manifest['source'] != batch['source'] or manifest['results_sha256'] != item['sha256']:
            raise ValueError('Historical record source or report checksum mismatch')
        if any(cfg.get(key) != value for key, value in base.items()):
            raise ValueError('Historical record physical config mismatch')
        with h5py.File(directory/'results.h5', 'r') as data:
            status = manifest['status']
            endpoint = float(data['final/t'][()])
            if data.attrs['status'] != status or not math.isfinite(endpoint):
                raise ValueError('HDF5 status or endpoint mismatch')
            if status == 'complete':
                if not math.isclose(endpoint, base['T'], rel_tol=0, abs_tol=1e-9):
                    raise ValueError('Complete input has an incompatible endpoint')
                requested = data['snapshots/requested_times'][:]
                actual = data['snapshots/actual_times'][:]
                if (len(requested) != len(times) or len(actual) != len(times)
                        or any(not math.isclose(float(t), expected, rel_tol=0, abs_tol=1e-9)
                               for seq in (requested, actual) for t, expected in zip(seq, times))):
                    raise ValueError('Complete input is missing observation nodes')
            elif not 0 <= endpoint < base['T']:
                raise ValueError('Failed input has no valid accepted prefix')
        records[saved.name] = cfg, manifest
    references = batch['references']
    for key, offset in [('reference', 0), ('refined', 1)]:
        cfg, manifest = records[Path(references[key]).name]
        expected_dt = batch['config']['tau_base']*2**(-references['reference_k']-offset)
        if (manifest['status'] != 'complete' or cfg['scheme'] != 'sdirk2'
                or not math.isclose(cfg['dt'], expected_dt, rel_tol=1e-12)):
            raise ValueError('Reference pair is incompatible with captured settings')
    with table.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    trials = {Path(t['path']).name: t for t in batch['trials']}
    if (len(trials) != len(batch['trials'])
            or {t['scheme'] for t in trials.values()} != {'sdirk2', 'sdirk2_mrsav'}):
        raise ValueError('Expected one SDIRK2 comparison pair without duplicate trials')
    if set(records) != set(trials) | {Path(references[k]).name for k in ('reference', 'refined')}:
        raise ValueError('Report input list does not cover the captured study')
    seen = set()
    for row in rows:
        name = row['run_id']
        trial = trials[name]
        cfg, manifest = records[name]
        expected_dt = batch['config']['tau_base']*2**(-trial['k'])
        if not math.isclose(cfg['dt'], expected_dt, rel_tol=1e-12):
            raise ValueError('Trial time step does not match the captured refinement level')
        observation = name, float(row['time'])
        if observation in seen or observation[1] not in times:
            raise ValueError('Duplicate or unexpected observation row')
        seen.add(observation)
        if (row['scheme'] != cfg['scheme'] or row['scheme'] != trial['scheme']
                or float(row['k']) != trial['k'] or row['status'] != manifest['status']
                or not math.isclose(float(row['dt']), cfg['dt'], rel_tol=1e-12)):
            raise ValueError('Saved table and trial identity mismatch')
        if row['status'] == 'complete' and any(not math.isfinite(float(row[n])) or float(row[n]) < 0
                                                for n in ('L2', 'H1')):
            raise ValueError('Complete trial has invalid saved errors')
    if seen != {(name, t) for name in trials for t in times}:
        raise ValueError('Saved error table is incomplete')
    physical = records[next(iter(trials))][0]
    for name in records:
        if any(records[name][0].get(k) != physical.get(k) for k in ('force', 'initial_condition', 'boundary')):
            raise ValueError('Mixed forcing, initial condition or boundary identities')
    return batch, physical, sorted(rows, key=lambda r: (float(r['time']), r['scheme'], float(r['k'])))

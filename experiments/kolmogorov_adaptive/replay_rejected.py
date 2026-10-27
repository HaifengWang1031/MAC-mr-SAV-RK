"""Recover rejected velocity embedded differences from saved MAC checkpoints.

Replay only recorded attempts between the nearest preceding saved state and
the last rejection in each interval. Keep model, step sizes, acceptance flags
and original source fixed. Validate replayed indicators and accepted energies;
write a separate diagnostic run without changing original data or timings.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--member', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    from experiments.kolmogorov_adaptive.search_tolerance import _restart_single_thread
    _restart_single_thread()

import numpy as np

from experiments.workflow import PROJECT, digest, provenance, write_json
from experiments.kolmogorov_adaptive.analyze import read_member
from experiments.kolmogorov_adaptive.run import make_scheme, model_and_initial
from solver.adaptivity.controller import ScalarErrorControl


def replay_intervals(data: dict[str, np.ndarray]) -> list[tuple[int, int, int]]:
    """Return (checkpoint index, first attempt, last rejected attempt)."""
    t = data['attempt_t']
    if np.any(np.diff(t) < 0):
        raise ValueError('Attempt start times are not ordered')
    rejected = np.flatnonzero(~data['attempt_accepted'].astype(bool))
    checkpoints = data['output_times']
    groups: dict[int, int] = {}
    for i in rejected:
        checkpoint = int(np.searchsorted(checkpoints, t[i], side='right')-1)
        if checkpoint < 0:
            raise ValueError('No saved state precedes a rejected attempt')
        groups[checkpoint] = int(i)
    result = []
    for checkpoint, last in sorted(groups.items()):
        first = int(np.searchsorted(t, checkpoints[checkpoint], side='left'))
        if first > last or t[first] != checkpoints[checkpoint]:
            raise ValueError('Saved checkpoint is not an attempt start state')
        result.append((checkpoint, first, last))
    return result


def replay(member: Path, *, root: Path = PROJECT) -> Path:
    config, manifest, data = read_member(member)
    source = provenance()
    if manifest['status'] != 'complete' or manifest['source'] != source:
        raise ValueError('Complete member and identical numerical source/environment required')
    if config['scheme'] != 'sdirk3_mrsav' or config.get('fixed_step') is not None:
        raise ValueError('An adaptive SDIRK3-mrSAV member is required')
    intervals = replay_intervals(data)
    if not intervals:
        raise ValueError('No rejected trials to replay')
    directory = root/'runs/kolmogorov_adaptive/rejected_replay'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    metadata: dict[str, Any] = {
        'status': 'running', 'member': str(member.resolve()),
        'result_sha256': manifest['result_sha256'], 'config_sha256': digest(member/'config.json'),
        'source': source, 'numerical_source_root': str(PROJECT),
        'entrypoint_sha256': digest(Path(__file__)),
        'validation_tolerances': {'rtol': 1e-9, 'atol': 1e-11},
        'intervals': intervals, 'purpose': 'Recover rejected unnormalized velocity L2 embedded differences'}
    write_json(directory/'manifest.json', metadata)
    write_json(directory/'config.json', config)
    checks: dict[str, float] = {}
    rows: list[dict[str, float | int]] = []
    start = perf_counter()
    with (directory/'run.log').open('w', buffering=1) as stream:
        def log(message: str) -> None:
            print(message, flush=True)
            stream.write(message+'\n')

        def check(name: str, actual: float, expected: float) -> None:
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11,
                                       err_msg=f'Replay mismatch: {name}')
            discrepancy = abs(actual-expected)/(1+abs(expected))
            checks[name] = max(checks.get(name, 0.), discrepancy)

        log(f'RUN={directory} intervals={len(intervals)}')
        try:
            model, _ = model_and_initial(config)
            scheme = make_scheme(config['scheme'], config['gamma'])
            scalar = ScalarErrorControl(atol=config['atol_scalar'], gamma=config['gamma'])
            node_indices = np.cumsum(data['attempt_accepted'].astype(int))
            count = 0
            for checkpoint, first, last in intervals:
                velocity = model.grid.pack(data['output_u'][checkpoint], data['output_v'][checkpoint])
                state = model.state(float(data['output_times'][checkpoint]), velocity,
                                    float(data['output_r'][checkpoint]))
                for i in range(first, last+1):
                    check('start_time', state.t, float(data['attempt_t'][i]))
                    step = float(data['attempt_h'][i])
                    trial = scheme.step(model, state, step)
                    if trial.embedded_velocity is None:
                        raise ValueError('Missing embedded velocity')
                    old, high = model.vector(state), model.vector(trial.state)
                    difference = model.grid.norm(high-trial.embedded_velocity)
                    old_norm, trial_norm = model.grid.norm(old), model.grid.norm(high)
                    scale = config['atol_velocity']+config['rtol_velocity']*max(
                        old_norm, trial_norm)
                    error_u = difference/scale
                    defect = scalar.defect(state.r, trial.state.r, step)
                    error_r = abs(defect)/scalar.atol
                    check('error_u', error_u, float(data['attempt_error_u'][i]))
                    check('error_r', error_r, float(data['attempt_error_r'][i]))
                    check('joint_error', max(error_u, error_r), float(data['attempt_error'][i]))
                    if bool(data['attempt_accepted'][i]) != (max(error_u, error_r) <= 1):
                        raise ValueError('Replay acceptance differs from stored decision')
                    if data['attempt_accepted'][i]:
                        state = trial.state
                        node = node_indices[i]
                        check('accepted_kinetic', .5*model.inner(high, high),
                              float(data['diagnostic_kinetic'][node]))
                        check('accepted_r', state.r, float(data['diagnostic_r'][node]))
                    else:
                        rows.append({'attempt_index': i, 'start_time': state.t, 'step': step,
                                     'trial_time': state.t+step,
                                     'velocity_l2_embedded_difference': difference,
                                     'old_velocity_l2': old_norm, 'trial_velocity_l2': trial_norm,
                                     'normalization_scale': scale, 'error_u': error_u,
                                     'error_r': error_r, 'scalar_defect': defect})
                    count += 1
                log(f'PROGRESS checkpoint={checkpoint} replayed={count} rejected={len(rows)} '
                    f'wall_s={perf_counter()-start:.1f}')
            expected = np.flatnonzero(~data['attempt_accepted'].astype(bool))
            np.testing.assert_array_equal([row['attempt_index'] for row in rows], expected)
            if digest(member/'results.npz') != manifest['result_sha256']:
                raise ValueError('Original results changed during replay')
            arrays: dict[str, Any] = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            np.savez_compressed(directory/'rejected_difference.npz', **arrays)
            metadata.update(status='complete', rejected_trials=len(rows), replayed_attempts=count,
                            max_scaled_discrepancies=checks,
                            diagnostic_sha256=digest(directory/'rejected_difference.npz'))
        except Exception as exc:
            metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            metadata['wall_seconds'] = perf_counter()-start
            write_json(directory/'manifest.json', metadata)
            log(f'FINAL status={metadata["status"]}')
    return directory


if __name__ == '__main__':
    print(replay(args.member, root=args.output_root))

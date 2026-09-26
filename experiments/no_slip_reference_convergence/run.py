"""Four-scheme, no-slip reference-solution time-convergence campaign."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.workflow import PROJECT, provenance, run_experiment, write_json
from experiments.no_slip_reference_convergence.analyze import analyze_batch

SCHEMES = ('sdirk2', 'sdirk2_mrsav', 'sdirk3', 'sdirk3_mrsav')


def schedule(times: list[float], nominal_step: float) -> list[float]:
    """Hit every observation time exactly; only interval-end steps are shortened."""
    steps: list[float] = []
    previous = 0.
    for endpoint in times:
        interval = endpoint-previous
        count = int(np.floor(interval/nominal_step))
        if count:
            steps.extend([nominal_step]*count)
        remainder = interval-count*nominal_step
        if remainder > 1e-12*interval:
            steps.append(remainder)
        else:
            steps[-1] += remainder
        previous = endpoint
    return steps


def validate(config: dict) -> None:
    base = config['base']
    levels = config['k_levels']
    times = base['snapshots']
    if base['experiment'] != 'forced_ns' or base['ic_kind'] != 'isotropic_beams' \
            or base['m'] != 1 or base['T'] != times[-1]:
        raise ValueError('Expected the forced, multi-mode no-slip benchmark')
    if not levels or not all(np.isfinite(levels)) or any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError('k_levels must be finite, distinct and increasing')
    if not times or any(b <= a for a, b in zip([0.]+times[:-1], times)):
        raise ValueError('Observation times must increase from zero')
    if config['reference_k'] <= levels[-1] or config['tau_base'] <= 0:
        raise ValueError('Reference step must be finer than every trial step')
    if not 0 < config['sensitivity_threshold'] < 1:
        raise ValueError('Invalid reference sensitivity threshold')


def run_campaign(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    validate(config)
    folder = root/'runs/no_slip_reference_convergence/batches'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    folder.mkdir(parents=True)
    manifest = folder/'batch.json'
    batch: dict = {'status': 'running', 'config': config, 'source': provenance(),
                   'members': [], 'references': {}, 'report': None}
    write_json(manifest, batch)
    with (folder/'run.log').open('w', buffering=1) as log:
        def message(value: str) -> None:
            log.write(value+'\n')
            print(value, flush=True)

        message(f'batch={manifest.resolve()}')

        def compute(scheme: str, level: float, role: str) -> Path:
            step = config['tau_base']*2.**(-level)
            message(f'start role={role} k={level:g} scheme={scheme} tau={step:.12g}')
            path = run_experiment({**config['base'], 'scheme': scheme,
                                   'steps': schedule(config['base']['snapshots'], step)},
                                  root=root, rerun=rerun)
            status = json.loads((path/'manifest.json').read_text())['status']
            message(f'{status} {path}')
            return path.resolve()

        try:
            for level in config['k_levels']:
                for scheme in SCHEMES:
                    path = compute(scheme, level, 'trial')
                    batch['members'].append({'k': level, 'scheme': scheme, 'path': str(path),
                                             'status': json.loads((path/'manifest.json').read_text())['status']})
                    write_json(manifest, batch)
            for key, level in [('reference', config['reference_k']),
                               ('refined', config['reference_k']+1)]:
                path = compute('sdirk3', level, key)
                batch['references'][key] = str(path)
                write_json(manifest, batch)
            report = analyze_batch(manifest, root=root)
            batch['report'] = str(report.resolve())
            analysis = json.loads((report/'analysis.json').read_text())
            failed = [m for m in batch['members'] if m['status'] != 'complete']
            batch['status'] = ('reference_unresolved' if not analysis['reference_passed'] else
                               'complete_with_failed_trials' if failed else 'complete')
        except Exception as exc:
            batch.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            message(batch['error'])
            raise
        finally:
            batch['finished_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(manifest, batch)
            message(f'FINAL status={batch["status"]} batch={manifest} report={batch["report"]}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    print(run_campaign(json.loads(args.config.read_text()), root=args.root, rerun=args.rerun))

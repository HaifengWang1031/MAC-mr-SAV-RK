"""Compare eight T=30 adaptive runs with common step and growth limits.

Purpose: test whether a shared maximum step of 0.005 and controller factor
range [0.5, 1.2] retain the selected methods' accuracy against the existing
same-grid SDIRK3 reference and fixed-step 0.0025 controls.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import process_time
from uuid import uuid4

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.workflow import PROJECT, provenance, write_json
from experiments.kolmogorov_adaptive.run import (
    config_check, make_scheme, model_and_initial, output_times, save_member,
)
from solver.adaptivity import IController, PIController, integrate_adaptive
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup

SCHEMES = ('sdirk2', 'sdirk2_mrsav', 'sdirk3', 'sdirk3_mrsav')
CONTROLLERS = {'I': IController, 'PI': PIController}


def load_inputs(path: Path, smoke: bool):
    specification = json.loads(path.read_text())
    selection = json.loads((PROJECT / specification['selection_record']).read_text())
    base = json.loads(Path(selection['base_batch']).read_text())
    config = {**base['config']}
    if smoke:
        config['T'] = 0.2
    config.update(initial_step=specification['common']['initial_step'],
                  min_step=specification['common']['min_step'],
                  max_step=specification['common']['max_step'],
                  atol_velocity=specification['common']['atol_velocity'])
    config_check(config)
    if base['status'] != 'complete' or selection['status'] != 'complete':
        raise ValueError('Baseline batch and selected controls must be complete')
    reference = Path(selection['reference'])
    manifest = json.loads((reference / 'manifest.json').read_text())
    reference_config = json.loads((reference / 'config.json').read_text())
    if manifest['status'] != 'complete' or hashlib.sha256(
            (reference / 'results.npz').read_bytes()).hexdigest() != manifest['result_sha256']:
        raise ValueError('Reference manifest or checksum mismatch')
    for key in ('nx', 'ny', 'nu', 'm', 'epsilon', 'initial_modes', 'gamma',
                'output_every', 'reference_step', 'stokes_backend',
                'stokes_tolerance', 'stokes_krylov_tolerance', 'stokes_max_iterations'):
        if reference_config[key] != config[key]:
            raise ValueError(f'Reference configuration differs at {key}')
    if reference_config['T'] < config['T']:
        raise ValueError('Reference duration is too short')
    return specification, selection, config, reference


def run(path: Path, smoke: bool) -> Path:
    specification, selection, config, reference = load_inputs(path, smoke)
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
    directory = PROJECT / 'runs/kolmogorov_adaptive/common_step_limits' / identity
    directory.mkdir(parents=True)
    record_path = directory / 'campaign.json'
    source = provenance()
    record = {
        'status': 'running', 'purpose': __doc__.strip(), 'smoke': smoke,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': source, 'specification': specification, 'base_config': config,
        'reference': str(reference), 'selection_record': specification['selection_record'],
        'fixed_0025_max_errors': selection['fixed_0025_max_errors'], 'members': [],
    }
    write_json(record_path, record)
    print(f'CAMPAIGN={record_path}', flush=True)
    outputs = output_times(config)
    with np.load(reference / 'results.npz') as data:
        reference_times = data['output_times'][:len(outputs)].copy()
        reference_u = data['output_u'][:len(outputs)].copy()
        reference_v = data['output_v'][:len(outputs)].copy()
    # The saved reference can drift by about 6e-11 over 30 units from
    # repeated floating-point time additions; indices still correspond.
    if not np.allclose(reference_times, outputs, rtol=0, atol=1e-9):
        record.update(status='failed', error='Reference output times do not match')
        write_json(record_path, record)
        raise ValueError(record['error'])
    grid = MACGrid(config['nx'], config['ny'], 2*np.pi, 2*np.pi)
    reference_norm = [max(grid.norm(grid.pack(u, v)), 1e-14)
                      for u, v in zip(reference_u, reference_v)]
    warmup()
    with (directory / 'run.log').open('w', buffering=1) as logfile:
        def log(message):
            print(message, flush=True)
            logfile.write(message + '\n')

        log(f'campaign={record_path}; reference={reference}; smoke={smoke}')
        log(f'common max_step={config["max_step"]}, min_factor={specification["common"]["min_factor"]}, '
            f'max_factor={specification["common"]["max_factor"]}')
        try:
            for scheme in SCHEMES[:1] if smoke else SCHEMES:
                for name, cls in CONTROLLERS.items():
                    selected = selection['selected'][f'{scheme}-{name}']['parameters']
                    model, initial = model_and_initial(config)
                    controller = cls(
                        atol=config['atol_velocity'], rtol=selected['rtol'],
                        safety=selected['safety'], min_step=config['min_step'],
                        max_step=config['max_step'],
                        min_factor=specification['common']['min_factor'],
                        max_factor=specification['common']['max_factor'],
                        estimator_order=2 if scheme.startswith('sdirk2') else 3,
                    )
                    label = f'{scheme}-{name}'
                    log(f'START {label} rtol={selected["rtol"]} safety={selected["safety"]}')
                    start_cpu = process_time()
                    result = integrate_adaptive(
                        model, make_scheme(scheme, config['gamma']), initial,
                        config['T'], config['initial_step'], controller=controller,
                        snapshots=outputs, strict_snapshots=True,
                        progress=lambda count, t, elapsed: log(
                            f'PROGRESS {label} steps={count} t={t:.6f}/{config["T"]} elapsed_s={elapsed:.1f}')
                        if count == 1 or count % config.get('progress_every', 1000) == 0 else None,
                    )
                    cpu = process_time() - start_cpu
                    member_path = directory / label
                    effective = {**config, 'rtol_velocity': selected['rtol'],
                                 'safety': selected['safety'],
                                 'min_factor': controller.min_factor,
                                 'max_factor': controller.max_factor,
                                 'estimator_order': controller.estimator_order}
                    save_member(member_path, effective, scheme, name, result, source)
                    entry = {
                        'scheme': scheme, 'controller': name, 'path': str(member_path),
                        'status': result.status, 'error': result.error,
                        'integration_cpu_seconds': cpu,
                        'accepted_steps': len(result.times)-1,
                        'rejected_trials': sum(not item['accepted'] for item in result.attempts),
                    }
                    if result.status == 'complete' and len(result.snapshots) == len(outputs):
                        errors = [grid.norm(grid.pack(state.u-u, state.v-v))/norm
                                  for state, u, v, norm in zip(
                                      result.snapshots, reference_u, reference_v, reference_norm)]
                        steps = np.asarray([item['step'] for item in result.attempts
                                            if item['accepted']])
                        ratios = np.maximum(steps[1:]/steps[:-1], steps[:-1]/steps[1:])
                        entry.update(max_relative_l2=float(max(errors)),
                                     final_relative_l2=float(errors[-1]),
                                     maximum_accepted_step=float(max(steps)),
                                     adjacent_step_ratio_p95=float(np.quantile(ratios, 0.95)),
                                     adjacent_step_ratio_max=float(max(ratios)))
                        if not smoke:
                            entry['better_than_fixed_0025'] = (
                                entry['max_relative_l2'] <
                                selection['fixed_0025_max_errors'][scheme])
                    record['members'].append(entry)
                    write_json(record_path, record)
                    log(f'END {label} status={entry["status"]} cpu_s={cpu:.2f} '
                        f'max_error={entry.get("max_relative_l2")} '
                        f'fixed_0025_better={entry.get("better_than_fixed_0025")}')
            record['status'] = ('complete' if all(item['status'] == 'complete'
                              for item in record['members']) else 'complete_with_failures')
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            log(f'ERROR {record["error"]}')
            raise
        finally:
            write_json(record_path, record)
            log(f'FINAL status={record["status"]} campaign={record_path}')
    return record_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    options = parser.parse_args()
    run(options.config, options.smoke)

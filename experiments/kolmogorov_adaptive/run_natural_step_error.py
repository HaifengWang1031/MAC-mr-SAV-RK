"""Eight natural-grid adaptive runs with optional per-trajectory reference errors.

The reference is a diagnostic only: it never changes the adaptive controller,
and its CPU time is removed from the reported adaptive integration time.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
from uuid import uuid4

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.workflow import PROJECT, provenance, write_json
from experiments.kolmogorov_adaptive.reference_error import ReferenceErrorObserver
from experiments.kolmogorov_adaptive.run import config_check, make_scheme, model_and_initial, save_member
from solver.adaptivity import IController, PIController, integrate_adaptive
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup

SCHEMES = ('sdirk2', 'sdirk2_mrsav', 'sdirk3', 'sdirk3_mrsav')
CONTROLLERS = {'I': IController, 'PI': PIController}


def resolved_inputs(path: Path, smoke: bool) -> tuple[dict, dict, dict]:
    specification = json.loads(path.read_text())
    config = json.loads((PROJECT / specification['base_config']).read_text())
    config.update(specification['common'])
    if smoke:
        config['T'] = 0.2
    config_check(config)
    tuned = json.loads((PROJECT / specification['tuned_parameters']).read_text())
    diagnostic = specification['reference_diagnostic']
    if not isinstance(diagnostic['enabled'], bool) or diagnostic['max_step'] <= 0 or \
            diagnostic['min_substeps'] < 2 or not 0 < diagnostic['verification_fraction'] < 1:
        raise ValueError('Invalid reference diagnostic configuration')
    cases = specification.get('cases')
    if cases is not None:
        if not isinstance(cases, list) or not cases:
            raise ValueError('cases must be a nonempty list')
        identities = set()
        for case in cases:
            if not isinstance(case, dict) or set(case) != {'scheme', 'controller', 'rtol'} or \
                    case['scheme'] not in SCHEMES or case['controller'] not in CONTROLLERS or \
                    not np.isfinite(case['rtol']) or case['rtol'] <= 0:
                raise ValueError('Invalid adaptive case')
            identity = case['scheme'], case['controller']
            if identity in identities:
                raise ValueError(f'Duplicate adaptive case: {identity}')
            identities.add(identity)
    return specification, config, tuned


def run(path: Path, smoke: bool = False) -> Path:
    specification, config, tuned = resolved_inputs(path, smoke)
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory = PROJECT/'runs/kolmogorov_adaptive/natural_step_error'/identity
    directory.mkdir(parents=True)
    record_path = directory/'campaign.json'
    source = provenance()
    record = {
        'status': 'running', 'purpose': __doc__.strip(), 'created_utc':
        datetime.now(timezone.utc).isoformat(), 'source': source,
        'smoke': smoke, 'specification': specification, 'base_config': config,
        'members': [],
    }
    write_json(record_path, record)
    print(f'CAMPAIGN={record_path}', flush=True)
    diagnostic = specification['reference_diagnostic']
    grid = MACGrid(config['nx'], config['ny'], 2*np.pi, 2*np.pi)
    with (directory/'run.log').open('w', buffering=1) as logfile:
        def log(message: str) -> None:
            print(message, flush=True)
            logfile.write(message+'\n')

        log(f'campaign={record_path} smoke={smoke} reference_error={diagnostic["enabled"]}')
        log(f'natural nodes: max_step={config["max_step"]} '
            f'factor_range=[{config["min_factor"]},{config["max_factor"]}]')
        try:
            warmup()
            cases = specification.get('cases')
            if cases is None:
                pairs = [(s, c, None) for s in SCHEMES for c in CONTROLLERS]
                if smoke:
                    pairs = [('sdirk2', 'I', None), ('sdirk3_mrsav', 'PI', None)]
            else:
                pairs = [(case['scheme'], case['controller'], case['rtol'])
                         for case in cases]
            for scheme_name, controller_name, rtol_override in pairs:
                label = f'{scheme_name}-{controller_name}'
                settings = dict(tuned['pairs'][scheme_name][controller_name])
                if rtol_override is not None:
                    settings['rtol'] = rtol_override
                model, initial = model_and_initial(config)
                cls = CONTROLLERS[controller_name]
                controller = cls(
                    atol=config['atol_velocity'], rtol=settings['rtol'],
                    safety=settings['safety'], min_step=config['min_step'],
                    max_step=config['max_step'], min_factor=config['min_factor'],
                    max_factor=config['max_factor'],
                    estimator_order=2 if scheme_name.startswith('sdirk2') else 3,
                )
                observer = None
                if diagnostic['enabled']:
                    coarse_model, coarse_initial = model_and_initial(config)
                    fine_model, fine_initial = model_and_initial(config)
                    observer = ReferenceErrorObserver(
                        grid, coarse_model, coarse_initial, fine_model, fine_initial,
                        max_step=diagnostic['max_step'],
                        min_substeps=diagnostic['min_substeps'],
                    )
                log(f'START {label} rtol={settings["rtol"]} safety={settings["safety"]}')
                start_cpu, start_wall = process_time(), perf_counter()
                result = integrate_adaptive(
                    model, make_scheme(scheme_name, config['gamma']), initial,
                    config['T'], config['initial_step'], controller=controller,
                    snapshots=None, strict_snapshots=False, on_accept=observer,
                    progress=lambda count, t, elapsed: log(
                        f'PROGRESS {label} steps={count} t={t:.6f}/{config["T"]} '
                        f'adaptive_elapsed_s={elapsed:.1f} '
                        f'reference_steps={0 if observer is None else observer.fine.steps}')
                    if count == 1 or count % config.get('progress_every', 1000) == 0 else None,
                )
                integration_cpu = process_time()-start_cpu-result.observer_cpu_seconds
                integration_wall = perf_counter()-start_wall-result.observer_wall_seconds
                member_path = directory/label
                effective = {**config, 'scheme': scheme_name, 'controller': controller_name,
                             'rtol_velocity': settings['rtol'], 'safety': settings['safety'],
                             'estimator_order': controller.estimator_order,
                             'compute_reference_error': diagnostic['enabled'],
                             'reference_diagnostic': diagnostic if diagnostic['enabled'] else None}
                save_member(member_path, effective, scheme_name, controller_name, result, source)
                entry = {
                    'scheme': scheme_name, 'controller': controller_name,
                    'path': str(member_path), 'status': result.status,
                    'error': result.error, 'accepted_steps': len(result.times)-1,
                    'rejected_trials': sum(not a['accepted'] for a in result.attempts),
                    'adaptive_cpu_seconds': integration_cpu,
                    'adaptive_wall_seconds': integration_wall,
                    'observer_cpu_seconds': result.observer_cpu_seconds,
                    'observer_wall_seconds': result.observer_wall_seconds,
                }
                accepted_steps = np.asarray([a['step'] for a in result.attempts if a['accepted']])
                entry['maximum_accepted_step'] = (float(accepted_steps.max())
                                                  if len(accepted_steps) else None)
                if observer is not None:
                    error_path = member_path/'reference_error.npz'
                    np.savez_compressed(error_path,
                                        times=np.asarray(observer.times),
                                        relative_velocity_l2=np.asarray(observer.velocity_error),
                                        relative_reference_difference=np.asarray(
                                            observer.reference_difference))
                    entry['reference_error_path'] = str(error_path)
                    entry['reference_error_sha256'] = hashlib.sha256(
                        error_path.read_bytes()).hexdigest()
                    entry['reference_cpu_seconds'] = (
                        observer.coarse.cpu_seconds+observer.fine.cpu_seconds)
                    entry['total_integration_cpu_seconds'] = (
                        integration_cpu+result.observer_cpu_seconds)
                    if result.status == 'complete' and len(observer.times) == len(result.times):
                        if not np.allclose(observer.times, result.times, rtol=0, atol=1e-10):
                            raise RuntimeError(f'Reference node mismatch for {label}')
                        entry.update(observer.summary(diagnostic['verification_fraction']))
                        if not entry['reference_verified']:
                            log(f'WARNING {label} reference refinement check did not pass')
                record['members'].append(entry)
                write_json(record_path, record)
                log(f'END {label} status={entry["status"]} '
                    f'adaptive_cpu_s={integration_cpu:.2f} '
                    f'reference_cpu_s={entry.get("reference_cpu_seconds", 0.):.2f} '
                    f'max_error={entry.get("max_relative_l2")} '
                    f'reference_verified={entry.get("reference_verified")}')
            all_complete = all(item['status'] == 'complete' for item in record['members'])
            all_verified = all(item.get('reference_verified', True) for item in record['members'])
            record['status'] = ('complete' if all_complete and all_verified else
                                'complete_with_unverified_reference' if all_complete else
                                'complete_with_failures')
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
    args = parser.parse_args()
    run(args.config, args.smoke)

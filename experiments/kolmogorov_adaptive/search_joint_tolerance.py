"""Measure the SDIRK3-mrSAV I accuracy/cost tradeoff with joint u/r control.

Physics, grid, step bounds and controller factors are fixed. Two common ordinary
SDIRK3 references are saved once; analysis compares exact uniform output nodes.
The small pilot checks the workflow and cost, not long-time parameter quality.
"""
import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter, process_time
from typing import Any
from uuid import uuid4

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.kolmogorov_adaptive.search_tolerance import _restart_single_thread

if __name__ == '__main__':
    _restart_single_thread()

import numpy as np

from experiments.kolmogorov_adaptive.run import (
    config_check, model_and_initial, output_times, reference_schedule, save_member,
)
from experiments.records import PROJECT, digest, provenance, write_json
from solver.adaptivity import IController, ScalarErrorControl, integrate_adaptive
from solver.mac.kernels import warmup
from solver.schemes.sdirk3 import SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV


def resolve_scan(path: Path, *, pilot: bool = False) -> tuple[dict, dict, list[dict]]:
    spec = json.loads(path.read_text())
    base = json.loads((PROJECT/spec['base_config']).read_text())
    base.update(spec['common'])
    if pilot:
        base.update(spec['pilot']['common'])
        spec = {**spec, 'kind': 'pilot', 'repeats': 1}
    config_check(base)
    if spec['kind'] not in ('pilot', 'production') or type(spec['repeats']) is not int \
            or spec['repeats'] < (3 if spec['kind'] == 'production' else 1):
        raise ValueError('Production scans require at least three timing repeats')
    cases = []
    for rtol, atol_r in product(spec['rtol_u_values'], spec['atol_r_values']):
        cases.append({'atol_u': spec['atol_u'], 'rtol_u': rtol, 'atol_r': atol_r,
                      'role': 'joint'})
    anchor = spec['atol_u_sensitivity_anchor']
    cases.extend({'atol_u': value, **anchor, 'role': 'sensitivity'}
                 for value in spec['atol_u_sensitivity_values'])
    cases.append({**spec['velocity_only_baseline'], 'atol_r': None, 'role': 'baseline'})
    identities = set()
    for index, case in enumerate(cases):
        values = [case['atol_u'], case['rtol_u']]
        if case['atol_r'] is not None:
            values.append(case['atol_r'])
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Scan tolerances must be finite and positive')
        identity = case['atol_u'], case['rtol_u'], case['atol_r']
        if identity in identities:
            raise ValueError(f'Duplicate tolerance case: {identity}')
        identities.add(identity)
        case['id'] = f'case-{index:02d}'
    steps = spec['reference_steps']
    if len(steps) != 2 or not 0 < steps[1] < steps[0] < base['max_step'] \
            or not 0 < spec['verification_fraction'] < 1:
        raise ValueError('Two nested reference steps and a verification fraction are required')
    return spec, base, cases


def run_reference(config: dict, step: float, directory: Path, source: dict,
                  log: Callable[[str], None], *, query_times: list[float] | None = None) -> dict:
    """Stream a reference, retaining only requested fields and stage-check maxima."""
    directory.mkdir()
    effective = {**config, 'scheme': 'sdirk3', 'controller': 'fixed', 'reference_step': step}
    write_json(directory/'config.json', effective)
    manifest: dict[str, Any] = {'status': 'running', 'source': source, 'reference': True}
    write_json(directory/'manifest.json', manifest)
    model, state = model_and_initial(config)
    outputs = output_times(config) if query_times is None else query_times
    if not outputs or outputs[0] != 0. or any(
            right <= left for left, right in zip(outputs[:-1], outputs[1:])):
        raise ValueError('Reference queries must start at zero and increase strictly')
    saved = [state]
    count = 0
    maxima = {'stage_residual': 0., 'stage_divergence_inf': 0.}
    start_cpu, start_wall = process_time(), perf_counter()
    try:
        for target in outputs[1:]:
            for h in reference_schedule([state.t, target], step):
                trial = SDIRK3().step(model, state, h)
                state = trial.state
                if not np.isfinite(model.vector(state)).all():
                    raise FloatingPointError('Nonfinite reference velocity')
                count += 1
                maxima['stage_residual'] = max(maxima['stage_residual'],
                                               max(s.residual for s in trial.stages))
                maxima['stage_divergence_inf'] = max(maxima['stage_divergence_inf'],
                                                     max(s.divergence_inf for s in trial.stages))
                if count == 1 or count % config['reference_progress_every'] == 0:
                    log(f'REFERENCE step={step:g} count={count} t={state.t:.6f}/{config["T"]}')
            if abs(state.t-target) > 1e-10*max(1., abs(target)):
                raise RuntimeError('Reference output time mismatch')
            state = replace(state, t=target)
            saved.append(state)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    manifest.update(integration_cpu_seconds=process_time()-start_cpu,
                    integration_wall_seconds=perf_counter()-start_wall,
                    accepted_steps=count, **maxima)
    np.savez_compressed(directory/'results.npz', output_times=[s.t for s in saved],
                        output_u=np.stack([s.u for s in saved]),
                        output_v=np.stack([s.v for s in saved]), output_r=[s.r for s in saved])
    manifest['result_sha256'] = digest(directory/'results.npz')
    write_json(directory/'manifest.json', manifest)
    log(f'END REFERENCE step={step:g} status={manifest["status"]} '
        f'cpu_s={manifest["integration_cpu_seconds"]:.2f}')
    return {'path': str(directory), 'step': step, **manifest}


def run_case(config: dict, case: dict, repeat: int, directory: Path,
             source: dict, log: Callable[[str], None]) -> dict:
    effective = {**config, 'atol_velocity': case['atol_u'], 'rtol_velocity': case['rtol_u'],
                 'atol_scalar': case['atol_r'], 'error_indicator':
                 'velocity_only' if case['atol_r'] is None else 'max_velocity_scalar_defect',
                 'strict_snapshots': True}
    setup_start = process_time()
    model, initial = model_and_initial(effective)
    setup_cpu = process_time()-setup_start
    control = IController(atol=case['atol_u'], rtol=case['rtol_u'],
                          safety=config['safety'], min_factor=config['min_factor'],
                          max_factor=config['max_factor'], min_step=config['min_step'],
                          max_step=config['max_step'])
    scalar = None if case['atol_r'] is None else ScalarErrorControl(case['atol_r'], config['gamma'])
    label = f'{case["id"]} repeat={repeat}'
    log(f'START {label} atol_u={case["atol_u"]:g} rtol_u={case["rtol_u"]:g} '
        f'atol_r={case["atol_r"]}')
    start_cpu = process_time()
    result = integrate_adaptive(
        model, SDIRK3MRSAV(config['gamma']), initial, config['T'], config['initial_step'],
        controller=control, scalar_control=scalar, snapshots=output_times(config),
        strict_snapshots=True, progress=lambda count, t, elapsed: log(
            f'PROGRESS {label} accepted={count} t={t:.6f}/{config["T"]} wall_s={elapsed:.1f}')
        if count == 1 or count % config['progress_every'] == 0 else None)
    cpu = process_time()-start_cpu
    save_member(directory, effective, 'sdirk3_mrsav', 'I', result, source)
    accepted = [a for a in result.attempts if a['accepted']]
    stages = [s for step in result.stages for s in step]
    record = {'case_id': case['id'], 'repeat': repeat, 'path': str(directory),
              'status': result.status, 'error': result.error, 'setup_cpu_seconds': setup_cpu,
              'integration_cpu_seconds': cpu, 'integration_wall_seconds': result.seconds,
              'accepted_steps': len(accepted), 'rejected_trials': len(result.attempts)-len(accepted),
              'max_abs_r': max(abs(d['r']) for d in result.diagnostics),
              'max_step_fraction': sum(np.isclose(a['step'], config['max_step'], rtol=1e-10)
                                       for a in accepted)/max(1, len(accepted)),
              'scalar_limited_fraction': (sum(float(a['error_r'])**(-.5) < float(a['error_u'])**(-1/3)
                                             for a in accepted if float(a['error_r']) > 0
                                             and float(a['error_u']) > 0)/max(1, len(accepted))
                                          if scalar is not None else None),
              'max_stage_residual': max((s['residual'] for s in stages), default=0.),
              'max_stage_divergence_inf': max((s['divergence_inf'] for s in stages), default=0.),
              'max_scalar_residual': max((s['scalar_residual'] for s in stages), default=0.)}
    write_json(directory/'timing.json', record)
    log(f'END {label} status={result.status} cpu_s={cpu:.2f} accepted={len(accepted)} '
        f'rejected={record["rejected_trials"]} max_abs_r={record["max_abs_r"]:.6g}')
    return record


def run_scan(path: Path, *, root: Path = PROJECT, pilot: bool = False) -> Path:
    spec, config, cases = resolve_scan(path, pilot=pilot)
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory = root/'runs/kolmogorov_adaptive/joint_tolerance_scan'/identity
    directory.mkdir(parents=True)
    campaign = directory/'campaign.json'
    source = provenance()
    record: dict[str, Any] = {'status': 'running', 'kind': spec['kind'], 'specification': spec,
                              'base_config': config, 'cases': cases, 'source': source,
                              'members': [], 'references': [], 'purpose': __doc__.strip()}
    write_json(campaign, record)
    with (directory/'run.log').open('w', buffering=1) as logfile:
        def log(message: str) -> None:
            print(message, flush=True)
            logfile.write(message+'\n')
        log(f'CAMPAIGN={campaign} kind={spec["kind"]} T={config["T"]} '
            f'cases={len(cases)} repeats={spec["repeats"]}')
        try:
            warmup_start = process_time()
            warmup()
            # Warm stage solves and estimator paths once, outside all timings.
            warm_model, warm_initial = model_and_initial(config)
            SDIRK3MRSAV(config['gamma']).step(warm_model, warm_initial, config['initial_step'])
            SDIRK3().step(warm_model, warm_initial, config['initial_step'])
            record['warmup_cpu_seconds'] = process_time()-warmup_start
            del warm_model, warm_initial
            for index, step in enumerate(spec['reference_steps']):
                reference = run_reference(config, step, directory/f'reference-{index}', source, log)
                record['references'].append(reference)
                write_json(campaign, record)
                if reference['status'] != 'complete':
                    raise RuntimeError('Reference computation failed')
            for repeat in range(spec['repeats']):
                order = cases if repeat % 2 == 0 else cases[::-1]
                if repeat >= 2:
                    order = order[repeat:]+order[:repeat]
                for case in order:
                    member = run_case(config, case, repeat,
                                      directory/f'{case["id"]}-repeat-{repeat}', source, log)
                    record['members'].append(member)
                    write_json(campaign, record)
            record['status'] = ('complete' if all(m['status'] == 'complete'
                                                 for m in record['members']) else 'failed')
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            write_json(campaign, record)
            log(f'FINAL status={record["status"]} campaign={campaign}')
    return campaign


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--report', action='store_true', help='Analyze saved results after computation')
    args = parser.parse_args()
    campaign_path = run_scan(args.config, root=args.root, pilot=args.pilot)
    if args.report:
        from experiments.kolmogorov_adaptive.analyze_joint_tolerance import analyze_scan
        print(f'REPORT={analyze_scan(campaign_path, root=args.root)}', flush=True)

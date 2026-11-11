"""Bounded same-state adaptive search; fixed ordinary SDIRK references are separate jobs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, process_time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info

from experiments.controllers import controller_settings, make_controller
from experiments.records import digest, provenance, write_json
from experiments.torque_transition.mac_model import make_mac_model, mac_observables
from experiments.torque_transition.preflight import PHASE_DEGREES, THREAD_KEYS
from solver.adaptivity import ScalarErrorControl, integrate_adaptive
from solver.integrate import integrate
from solver.schemes.sdirk3 import SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV


def candidates() -> list[dict[str, Any]]:
    """One-factor scalar sweep followed by paired tolerance and controller probes."""
    base = {'atol': 1e-8, 'rtol': 1e-5, 'scalar_atol': 1e-5, 'kind': 'PI',
            'min_factor': .5, 'max_factor': 1.2, 'max_step': .05,
            'proportional': .7, 'integral': .4}
    rows = [{'name': 'baseline', **base}]
    for scalar in (1e-4, 1e-3, 1e-2):
        rows.append({**base, 'name': f'scalar-{scalar:g}', 'scalar_atol': scalar})
    for velocity in (1e-4, 1e-3, 1e-2):
        rows.append({**base, 'name': f'velocity-{velocity:g}',
                     'rtol': velocity, 'scalar_atol': 1e-3})
    common = {**base, 'rtol': 1e-3, 'scalar_atol': 1e-3}
    rows.extend([{**common, 'name': 'scalar-loose', 'scalar_atol': 1e-2},
                 {**common, 'name': 'fast-factors', 'min_factor': .2, 'max_factor': 2.},
                 {**common, 'name': 'PI-08-03', 'min_factor': .2, 'max_factor': 2.,
                  'proportional': .8, 'integral': .3},
                 {**common, 'name': 'I-fast', 'kind': 'I', 'min_factor': .2, 'max_factor': 2.},
                 {**common, 'name': 'cap-001', 'max_step': .01}])
    return rows


def run_job(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run <=2 physical time units from an explicit MAC seed, preserving all trial checks."""
    duration = float(config['duration'])
    if not 0 < duration <= 2:
        raise ValueError('Tuning duration must be in (0,2], not production')
    seed = Path(config['seed'])
    if digest(seed) != config['seed_sha256']:
        raise ValueError('Seed checksum mismatch')
    with np.load(seed) as saved:
        if str(saved['discretization']) != 'mac':
            raise ValueError('Requires MAC seed')
        cells, case, start_t = int(saved['nodes']), int(saved['case']), float(saved['t'])
        velocity, scalar = saved['velocity'].copy(), float(saved['r'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', config)
    model = make_mac_model(cells, PHASE_DEGREES[case])
    initial = model.state(start_t, velocity, scalar)
    gamma = 1000.
    # The common warm trial is never committed. Configuration/seed identity is explicit.
    scheme: Any = SDIRK3() if config['mode'] == 'reference' else SDIRK3MRSAV(gamma)
    scheme.step(model, initial, .001)
    cpu, wall = process_time(), perf_counter()
    progress_count = 0

    def progress(state: Any) -> None:
        nonlocal progress_count
        progress_count += 1
        if progress_count % 200 == 0:
            write_json(output/'progress.json', {'t': state.t, 'steps': progress_count,
                                               'wall_seconds': perf_counter()-wall})

    settings = config.get('settings', {})
    if config['mode'] == 'reference':
        step = float(config['step'])
        if not 0 < step <= .01:
            raise ValueError('Reference step must be in (0,.01]')
        count = int(np.ceil(duration/step))
        if count > 20000:
            raise ValueError('Reference exceeds the bounded tuning step budget')
        schedule = np.full(count, duration/count)
        result = integrate(model, scheme, initial, schedule, on_accept=progress)
        attempts: list[dict[str, Any]] = []
        control = None
    elif config['mode'] == 'adaptive':
        options = {key: settings[key] for key in ('atol', 'rtol', 'kind', 'min_factor',
                                                  'max_factor', 'max_step')}
        if settings['kind'] == 'PI':
            options.update({key: settings[key] for key in ('proportional', 'integral')})
        options.update(min_step=1e-12, max_steps=50000)
        controller = make_controller(options, 'sdirk3_mrsav')
        result = integrate_adaptive(model, scheme, initial, duration, .001,
                                    controller=controller,
                                    scalar_control=ScalarErrorControl(settings['scalar_atol'], gamma),
                                    on_accept=progress)
        attempts = result.attempts
        control = controller_settings(controller)
    else:
        raise ValueError('Unknown mode')
    elapsed, cpu_seconds = perf_counter()-wall, process_time()-cpu
    vector = model.vector(result.final)
    np.savez(output/'final.npz', velocity=vector, r=result.final.r, t=result.final.t,
             nodes=cells, case=case, discretization='mac')
    np.savez(output/'history.npz', times=result.times,
             steps=[a['step'] for a in attempts], accepted=[a['accepted'] for a in attempts],
             error_u=[a['error_u'] for a in attempts], error_r=[a['error_r'] for a in attempts],
             errors=[a['error'] for a in attempts])
    stages = [stage for row in result.stages for stage in row]
    accepted = [a for a in attempts if a['accepted']]
    row = {'name': config['name'], 'status': result.status, 'error': result.error,
           'mode': config['mode'], 'source': provenance(), 'config_sha256': digest(output/'config.json'),
           'seed_sha256': config['seed_sha256'], 'grid_cells': cells, 'case': case,
           'start_t': start_t, 'final_t': result.final.t, 'duration': duration,
           'wall_seconds': elapsed, 'cpu_seconds': cpu_seconds,
           'accepted_steps': len(result.times)-1,
           'attempts': len(attempts) if attempts else len(result.times)-1,
           'controller': control, 'settings': settings,
           'scalar_dominant_fraction': float(np.mean([a['error_r'] > a['error_u'] for a in accepted]))
                                       if accepted else None,
           'max_error_u': max((a['error_u'] for a in accepted), default=0.),
           'max_error_r': max((a['error_r'] for a in accepted), default=0.),
           'max_coupling_deviation': max((abs(s['frozen_factor']*(1-s['r'])-1)
                                         for s in stages), default=0.) if attempts else 0.,
           'max_stage_residual': max((s['residual'] for s in stages), default=0.),
           'max_full_divergence': max(d['divergence_inf'] for d in result.diagnostics),
           'observables': mac_observables(model, result.final, case),
           'threadpools': threadpool_info(),
           'artifacts': {name: digest(output/name) for name in ('config.json', 'final.npz', 'history.npz')},
           'qualification': 'bounded_common_seed; no PI-history resume; not stationary or production'}
    write_json(output/'result.json', row)
    return row


def make_plan(seed: Path, output: Path, *, duration: float = .1,
              reference_step: float = .00025,
              settings: list[dict[str, Any]] | None = None, repeats: int = 1) -> None:
    """Freeze a bounded candidate matrix before timing; repeats interleave reverse order."""
    if not 0 < duration <= 2 or not 0 < reference_step <= .01 or duration/(reference_step/2) > 20000:
        raise ValueError('Invalid bounded tuning duration/reference budget')
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError('Use 1..5 repeats')
    selected = settings if settings is not None else candidates()
    names = [item['name'] for item in selected]
    if len(names) != len(set(names)) or any(not name or any(
            c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in name)
            for name in names):
        raise ValueError('Candidate names must be unique safe filenames')
    output.mkdir(parents=True, exist_ok=False)
    seed = seed.resolve()
    common = {'seed': str(seed), 'seed_sha256': digest(seed), 'duration': duration}
    jobs = [{**common, 'name': name, 'mode': 'reference', 'step': step}
            for name, step in (('reference', reference_step), ('reference-half', reference_step/2))]
    for repeat in range(repeats):
        order = selected if repeat % 2 == 0 else list(reversed(selected))
        jobs += [{**common, 'name': f'{item["name"]}-r{repeat}', 'mode': 'adaptive',
                  'settings': item} for item in order]
    write_json(output/'plan.json', {'schema': 1, 'jobs': jobs, 'source': provenance(),
                                   'workers_default': 6, 'production_approved': False,
                                   'gate': {'velocity_relative': 1e-3, 'observables_relative': .01,
                                            'reference_fraction': .05, 'coupling': 1e-3,
                                            'full_divergence': 1e-8}})


def execute(output: Path, workers: int) -> None:
    """Execute only the explicit frozen short-test plan, never search or launch production."""
    if not 1 <= workers <= 6:
        raise ValueError('Use 1..6 single-thread workers')
    plan = json.loads((output/'plan.json').read_text())
    if (output/'execution.json').exists():
        raise ValueError('Execution already exists; use a new plan directory')
    write_json(output/'execution.json', {'status': 'running', 'plan_sha256': digest(output/'plan.json'),
                                       'workers': workers})
    jobs = plan['jobs']
    for job in jobs:
        write_json(output/f'{job["name"]}.json', job)

    def run_subprocess(job: dict[str, Any]) -> dict[str, Any]:
        env = dict(os.environ, **{key: '1' for key in THREAD_KEYS})
        name = job['name']
        with (output/f'{name}.log').open('w') as log:
            completed = subprocess.run([sys.executable, '-m', 'experiments.torque_transition.tune',
                                        'job', '--config', str(output/f'{name}.json'),
                                        '--output', str(output/'jobs'/name)],
                                       env=env, stdout=log, stderr=log, timeout=1800)
        return {'name': name, 'exit_code': completed.returncode}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_subprocess, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(output/'progress.json', {'finished': rows, 'planned': len(jobs)})
            print(json.dumps(row), flush=True)
    write_json(output/'execution.json', {'status': 'complete' if all(r['exit_code'] == 0 for r in rows) else 'failed',
                                       'plan_sha256': digest(output/'plan.json'), 'workers': workers,
                                       'jobs': rows})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'execute', 'job'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--duration', type=float, default=.1)
    parser.add_argument('--reference-step', type=float, default=.00025)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--settings', type=Path, help='JSON list of explicit candidate settings')
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.mode == 'plan':
        if args.seed is None:
            parser.error('plan requires --seed')
        make_plan(args.seed, args.output, duration=args.duration, reference_step=args.reference_step,
                  settings=None if args.settings is None else json.loads(args.settings.read_text()),
                  repeats=args.repeats)
    elif args.mode == 'execute':
        execute(args.output, args.workers)
    else:
        if args.config is None:
            parser.error('job requires --config')
        run_job(json.loads(args.config.read_text()), args.output)

"""Interleaved single-thread CPU benchmark for adaptive velocity tolerances."""
import os
import sys

THREAD_VARIABLES = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS',
)
_THREAD_MARKER = 'MAC_MRSAV_SINGLE_THREAD_BENCHMARK'


def _restart_single_thread() -> None:
    if os.environ.get(_THREAD_MARKER) == '1':
        return
    environment = os.environ.copy()
    environment.update({name: '1' for name in THREAD_VARIABLES})
    environment[_THREAD_MARKER] = '1'
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


if __name__ == '__main__':
    _restart_single_thread()

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter, process_time
from typing import Any
from uuid import uuid4

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.kolmogorov_adaptive.run import (
    SCHEMES,
    SDIRK3,
    config_check,
    make_scheme,
    model_and_initial,
    output_times,
    reference_schedule,
)
from experiments.workflow import PROJECT, provenance, write_json
from solver.adaptivity import IController, PIController, integrate_adaptive
from solver.integrate import integrate
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup

CONTROLLERS = {'I': IController, 'PI': PIController}


def tolerance_config_check(config: dict) -> None:
    config_check(config)
    tolerances = config.get('tolerances')
    if not isinstance(tolerances, list) or len(tolerances) < 2 or \
            len(set(tolerances)) != len(tolerances) or \
            not np.isfinite(tolerances).all() or any(value <= 0 for value in tolerances):
        raise ValueError('At least two distinct positive tolerances are required')
    if type(config.get('repeats')) is not int or config['repeats'] < 3:
        raise ValueError('At least three timing repeats are required')
    if not np.isfinite(config.get('atol_rtol_ratio', np.nan)) or \
            config['atol_rtol_ratio'] <= 0:
        raise ValueError('atol_rtol_ratio must be positive')
    if not np.isfinite(config.get('target_max_relative_error', np.nan)) or \
            config['target_max_relative_error'] <= 0:
        raise ValueError('target_max_relative_error must be positive')
    if config.get('baseline_rtol') not in tolerances:
        raise ValueError('baseline_rtol must be one of tolerances')


def _require_single_thread() -> None:
    incorrect = {name: os.environ.get(name) for name in THREAD_VARIABLES
                 if os.environ.get(name) != '1'}
    if incorrect:
        raise RuntimeError(f'Tolerance timing requires one thread: {incorrect}')


def _case_order(config: dict, repeat: int) -> list[tuple[str, str, float]]:
    cases = list(product(SCHEMES, CONTROLLERS, config['tolerances']))
    if repeat % 3 == 1:
        return list(reversed(cases))
    if repeat % 3 == 2:
        shift = len(cases)//2
        return cases[shift:]+cases[:shift]
    return cases


def _relative_errors(grid: MACGrid, snapshots: list[Any], reference: list[Any]) -> list[float]:
    errors = []
    for state, target in zip(snapshots, reference, strict=True):
        velocity = grid.pack(state.u, state.v)
        target_velocity = grid.pack(target.u, target.v)
        errors.append(grid.norm(velocity-target_velocity) /
                      max(grid.norm(target_velocity), 1e-14))
    return errors


def _median_absolute_deviation(values: list[float]) -> float:
    array = np.asarray(values)
    median = np.median(array)
    return float(np.median(np.abs(array-median)))


def summarize_records(records: list[dict], target: float) -> tuple[list[dict], list[dict], dict | None]:
    groups: dict[tuple[str, str, float], list[dict]] = {}
    for record in records:
        groups.setdefault((record['scheme'], record['controller'], record['rtol']), []).append(record)
    summary = []
    for (scheme, controller, rtol), items in groups.items():
        complete = [item for item in items if item['status'] == 'complete']
        cpu = [item['integration_cpu_seconds'] for item in complete]
        errors = [item['max_relative_error'] for item in complete]
        eligible = len(complete) == len(items) and len(items) >= 3 and max(errors) <= target
        summary.append({
            'scheme': scheme,
            'controller': controller,
            'rtol': rtol,
            'atol': items[0]['atol'],
            'repeats': len(items),
            'complete_repeats': len(complete),
            'median_integration_cpu_seconds': None if not cpu else float(np.median(cpu)),
            'cpu_mad_seconds': None if not cpu else _median_absolute_deviation(cpu),
            'min_integration_cpu_seconds': None if not cpu else min(cpu),
            'max_integration_cpu_seconds': None if not cpu else max(cpu),
            'max_relative_error': None if not errors else max(errors),
            'eligible': eligible,
            'selected': False,
        })
    summary.sort(key=lambda item: (item['scheme'], item['controller'], -item['rtol']))
    recommendations = []
    for scheme, controller in product(SCHEMES, CONTROLLERS):
        eligible = [item for item in summary if item['scheme'] == scheme and
                    item['controller'] == controller and item['eligible']]
        selected = min(eligible, key=lambda item: (
            item['median_integration_cpu_seconds'], -item['rtol'])) if eligible else None
        if selected is not None:
            selected['selected'] = True
        recommendations.append({
            'scheme': scheme,
            'controller': controller,
            'rtol': None if selected is None else selected['rtol'],
            'atol': None if selected is None else selected['atol'],
            'median_integration_cpu_seconds': (
                None if selected is None else selected['median_integration_cpu_seconds']),
        })
    common_candidates = []
    for rtol in sorted({item['rtol'] for item in summary}, reverse=True):
        items = [item for item in summary if item['rtol'] == rtol]
        if len(items) == len(SCHEMES)*len(CONTROLLERS) and all(
                item['eligible'] for item in items):
            common_candidates.append({
                'rtol': rtol,
                'atol': items[0]['atol'],
                'aggregate_median_cpu_seconds': sum(
                    item['median_integration_cpu_seconds'] for item in items),
                'max_relative_error': max(item['max_relative_error'] for item in items),
            })
    common = min(common_candidates, key=lambda item: (
        item['aggregate_median_cpu_seconds'], -item['rtol'])) if common_candidates else None
    return summary, recommendations, common


def _warm_numerical_paths(config: dict) -> float:
    start = perf_counter()
    warmup()
    for scheme_name in SCHEMES:
        model, initial = model_and_initial(config)
        trial = make_scheme(scheme_name, config['gamma']).step(
            model, initial, config['initial_step'])
        if not np.isfinite(model.vector(trial.state)).all():
            raise FloatingPointError('Nonfinite numerical warmup')
    return perf_counter()-start


def run_tolerance_scan(config: dict, *, root: Path = PROJECT) -> Path:
    tolerance_config_check(config)
    _require_single_thread()
    source = provenance()
    identity = (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+
                source['code_sha256'][:8]+'-'+uuid4().hex[:8]+'-adaptive-tolerance-scan')
    directory = root/'reports/timings'/identity
    directory.mkdir(parents=True)
    report_path = directory/'timings.json'
    report: dict[str, Any] = {
        'status': 'running',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': source,
        'config': config,
        'primary_timing': 'process_time integration only, JIT and model setup excluded',
        'records': [],
    }
    write_json(report_path, report)
    try:
        report['numerical_warmup_wall_seconds'] = _warm_numerical_paths(config)
        outputs = output_times(config)
        setup_cpu_start, setup_wall_start = process_time(), perf_counter()
        reference_model, reference_initial = model_and_initial(config)
        reference_setup_cpu = process_time()-setup_cpu_start
        reference_setup_wall = perf_counter()-setup_wall_start
        cpu_start, wall_start = process_time(), perf_counter()
        reference = integrate(
            reference_model, SDIRK3(), reference_initial,
            reference_schedule(outputs, config['reference_step']), snapshots=outputs)
        if reference.status != 'complete' or len(reference.snapshots) != len(outputs):
            raise RuntimeError(f'Reference failed: {reference.error}')
        report['reference'] = {
            'step': config['reference_step'],
            'steps': len(reference.times)-1,
            'setup_cpu_seconds': reference_setup_cpu,
            'setup_wall_seconds': reference_setup_wall,
            'integration_cpu_seconds': process_time()-cpu_start,
            'integration_wall_seconds': perf_counter()-wall_start,
        }
        grid = MACGrid(config['nx'], config['ny'], 2*np.pi, 2*np.pi)
        for repeat in range(config['repeats']):
            for scheme_name, controller_name, rtol in _case_order(config, repeat):
                atol = config['atol_rtol_ratio']*rtol
                setup_cpu_start, setup_wall_start = process_time(), perf_counter()
                model, initial = model_and_initial(config)
                controller = CONTROLLERS[controller_name](
                    atol=atol,
                    rtol=rtol,
                    safety=config['safety'],
                    min_step=config['min_step'],
                    max_step=config['max_step'],
                    estimator_order=2 if scheme_name.startswith('sdirk2') else 3,
                )
                setup_cpu = process_time()-setup_cpu_start
                setup_wall = perf_counter()-setup_wall_start
                cpu_start, wall_start = process_time(), perf_counter()
                result = integrate_adaptive(
                    model, make_scheme(scheme_name, config['gamma']), initial,
                    config['T'], config['initial_step'], controller=controller,
                    snapshots=outputs, strict_snapshots=True)
                integration_cpu = process_time()-cpu_start
                integration_wall = perf_counter()-wall_start
                errors = (_relative_errors(grid, result.snapshots, reference.snapshots)
                          if len(result.snapshots) == len(reference.snapshots) else [])
                accepted = sum(bool(item['accepted']) for item in result.attempts)
                rejection_reasons = Counter(
                    str(item['reason']) for item in result.attempts if not item['accepted'])
                backend = model.backend
                record = {
                    'repeat': repeat+1,
                    'scheme': scheme_name,
                    'controller': controller_name,
                    'rtol': rtol,
                    'atol': atol,
                    'status': result.status,
                    'error': result.error,
                    'setup_cpu_seconds': setup_cpu,
                    'setup_wall_seconds': setup_wall,
                    'integration_cpu_seconds': integration_cpu,
                    'integration_wall_seconds': integration_wall,
                    'accepted_steps': accepted,
                    'rejected_trials': len(result.attempts)-accepted,
                    'rejection_reasons': dict(rejection_reasons),
                    'boundary_partition_trials': sum(
                        bool(item.get('boundary_partition')) for item in result.attempts),
                    'minimum_trial_step': min(
                        (float(item['step']) for item in result.attempts), default=None),
                    'final_relative_error': None if not errors else errors[-1],
                    'max_relative_error': None if not errors else max(errors),
                    'max_divergence': max(
                        float(item['divergence_inf']) for item in result.diagnostics),
                    'stokes_solve_seconds': getattr(backend, 'solve_seconds', None),
                    'stokes_solves': getattr(backend, 'solves', None),
                    'stokes_right_hand_sides': getattr(backend, 'right_hand_sides', None),
                    'stokes_iterations': getattr(backend, 'total_iterations', None),
                }
                report['records'].append(record)
                write_json(report_path, report)
                print(f'repeat={repeat+1} scheme={scheme_name} controller={controller_name} '
                      f'rtol={rtol:g} status={result.status} cpu={integration_cpu:.3f}s '
                      f'error={record["max_relative_error"]}', flush=True)
        summary, recommendations, common = summarize_records(
            report['records'], config['target_max_relative_error'])
        report['summary'] = summary
        report['recommendations'] = recommendations
        report['common_recommendation'] = common
        baseline = [item for item in summary if item['rtol'] == config['baseline_rtol']]
        if common is not None and len(baseline) == len(SCHEMES)*len(CONTROLLERS):
            baseline_cpu = sum(item['median_integration_cpu_seconds'] for item in baseline)
            common['cpu_speedup_vs_baseline'] = (
                baseline_cpu/common['aggregate_median_cpu_seconds'])
            common['cpu_reduction_percent_vs_baseline'] = 100*(
                1-common['aggregate_median_cpu_seconds']/baseline_cpu)
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(report_path, report)
    return report_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    arguments = parser.parse_args()
    print(run_tolerance_scan(json.loads(arguments.config.read_text()), root=arguments.root))

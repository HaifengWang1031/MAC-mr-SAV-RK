"""Natural-node joint SDIRK3-mrSAV versus explicit fixed-step controls.

Keep physics and spatial discretization fixed; compare reference errors, r and
integration CPU. Shared references visit the union of actual saved sample times,
so output sampling never clips the adaptive steps. Retain failed finite prefixes.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import process_time
from typing import Any
from uuid import uuid4

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.kolmogorov_adaptive.search_tolerance import _restart_single_thread

if __name__ == '__main__':
    _restart_single_thread()

import numpy as np

from experiments.kolmogorov_adaptive.run import config_check, model_and_initial, output_times, save_member
from experiments.kolmogorov_adaptive.search_joint_tolerance import run_reference
from experiments.records import PROJECT, digest, provenance, write_json
from solver.adaptivity import IController, ScalarErrorControl, integrate_adaptive
from solver.integrate import Result, integrate, step_sizes
from solver.mac.kernels import warmup
from solver.schemes.sdirk3 import SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV


def resolved_config(path: Path, *, pilot: bool = False) -> tuple[dict, dict]:
    spec = json.loads(path.read_text())
    config = json.loads((PROJECT/spec['base_config']).read_text())
    config.update(spec['common'])
    if pilot:
        config.update({k: v for k, v in spec['pilot'].items() if k != 'repeats'})
        spec = {**spec, 'repeats': spec['pilot']['repeats']}
    check = {**config, 'max_step': float('inf') if config['max_step'] is None else config['max_step']}
    config_check(check)
    IController(atol=config['atol_velocity'], rtol=config['rtol_velocity'],
                max_step=check['max_step'], min_step=config['min_step'])
    ScalarErrorControl(config['atol_scalar'], config['gamma'])
    fixed = spec['fixed_steps']
    references = spec['reference_steps']
    if not fixed or len(set(fixed)) != len(fixed) or not np.isfinite(fixed).all() \
            or min(fixed) <= 0 or len(references) != 2 or \
            not 0 < references[1] < references[0] < min(fixed) \
            or type(spec['repeats']) is not int or spec['repeats'] < 1 \
            or not 0 < spec['verification_fraction'] < 1:
        raise ValueError('Invalid fixed controls, nested references or timing repeats')
    return spec, config


def save_stage_checks(directory: Path, result: Result) -> None:
    count = len(result.stages)
    values: dict[str, Any] = {'final_u': result.final.u, 'final_v': result.final.v,
                              'final_r': result.final.r, 'final_t': result.final.t,
                              'output_requests': np.asarray(result.snapshot_requests)}
    for key in ('r', 'residual', 'divergence_inf', 'continuity_residual', 'scalar_residual'):
        values[key] = np.asarray([[s[key] for s in stages] for stages in result.stages]).reshape(count, 4)
    candidates = np.full((count, 4, 5), np.nan)
    residuals = np.full_like(candidates, np.nan)
    root_count = np.zeros((count, 4), dtype=int)
    for i, stages in enumerate(result.stages):
        for j, stage in enumerate(stages):
            n = len(stage['candidates'])
            root_count[i, j] = n
            candidates[i, j, :n] = stage['candidates']
            residuals[i, j, :n] = stage['root_residuals']
    values.update(candidates=candidates, root_residuals=residuals, root_count=root_count,
                  final_stage_pressure=np.stack([s.pressure for s in result.final_stages])
                  if result.final_stages else np.empty((0,)))
    np.savez_compressed(directory/'stage_checks.npz', **values)
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['stage_checks_sha256'] = digest(directory/'stage_checks.npz')
    write_json(directory/'manifest.json', manifest)


def reference_queries(times: list[float], config: dict) -> list[float]:
    """Merge only floating-point time-label drift, never output-align time steps."""
    queries = [0., config['T']]
    for time in times:
        nominal = config['output_every']*round(time/config['output_every'])
        queries.append(nominal if abs(time-nominal) < 1e-10 else time)
    merged: list[float] = []
    for time in sorted(queries):
        if not merged or time-merged[-1] > 1e-10:
            merged.append(float(time))
    return merged


def run(path: Path, *, root: Path = PROJECT, pilot: bool = False) -> Path:
    spec, config = resolved_config(path, pilot=pilot)
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory = root/'runs/kolmogorov_adaptive/adaptive_fixed'/identity
    directory.mkdir(parents=True)
    campaign_path = directory/'campaign.json'
    source = provenance()
    record: dict[str, Any] = {'status': 'running', 'kind': 'pilot' if pilot else 'production',
                              'specification': spec, 'config': config, 'source': source,
                              'members': [], 'references': [], 'purpose': __doc__.strip(),
                              'snapshot_policy': 'nearest natural node, actual times retained',
                              'max_step_encoding': 'null means infinity'}
    write_json(campaign_path, record)
    with (directory/'run.log').open('w', buffering=1) as logfile:
        def log(message: str) -> None:
            print(message, flush=True)
            logfile.write(message+'\n')
        log(f'CAMPAIGN={campaign_path} kind={record["kind"]} grid={config["nx"]}x{config["ny"]} '
            f'T={config["T"]} repeats={spec["repeats"]}')
        try:
            start = process_time()
            warmup()
            model, initial = model_and_initial(config)
            SDIRK3MRSAV(config['gamma']).step(model, initial, config['initial_step'])
            SDIRK3().step(model, initial, config['initial_step'])
            record['warmup_cpu_seconds'] = process_time()-start
            del model, initial
            cases: list[tuple[str, float | None]] = [('adaptive', None)]+[
                (f'fixed-{h:g}', h) for h in spec['fixed_steps']]
            query_times: list[float] = []
            for repeat in range(spec['repeats']):
                order = cases if repeat % 2 == 0 else cases[::-1]
                if repeat >= 2:
                    order = order[repeat:]+order[:repeat]
                for label, h in order:
                    setup = process_time()
                    model, initial = model_and_initial(config)
                    setup_cpu = process_time()-setup
                    cpu_history: list[float] = []
                    log(f'START {label} repeat={repeat}')
                    start = process_time()
                    def progress(count: int, t: float, elapsed: float) -> None:
                        cpu_history.append(process_time()-start)
                        if count == 1 or count % config['progress_every'] == 0:
                            log(f'PROGRESS {label} repeat={repeat} steps={count} '
                                f't={t:.6f}/{config["T"]} integration_wall_s={elapsed:.1f}')
                    scheme = SDIRK3MRSAV(config['gamma'])
                    result: Result
                    if h is None:
                        control = IController(
                            atol=config['atol_velocity'], rtol=config['rtol_velocity'],
                            min_step=config['min_step'], max_step=float('inf') if config['max_step']
                            is None else config['max_step'], safety=config['safety'],
                            min_factor=config['min_factor'], max_factor=config['max_factor'])
                        result = integrate_adaptive(
                            model, scheme, initial, config['T'], config['initial_step'],
                            controller=control, scalar_control=ScalarErrorControl(
                                config['atol_scalar'], config['gamma']), strict_snapshots=False,
                            snapshots=output_times(config), progress=progress)
                        # Future requests in a failed trajectory are not valid samples.
                        valid = [i for i, t in enumerate(result.snapshot_requests)
                                 if t <= result.final.t+1e-10]
                        result.snapshot_requests = [result.snapshot_requests[i] for i in valid]
                        result.snapshot_times = [result.snapshot_times[i] for i in valid]
                        result.snapshots = [result.snapshots[i] for i in valid]
                    else:
                        result = integrate(model, scheme, initial, step_sizes(config['T'], dt=h),
                                           snapshots=output_times(config), progress=progress)
                    cpu = process_time()-start
                    if not result.snapshot_times or abs(result.snapshot_times[-1]-result.final.t) > 1e-10:
                        result.snapshot_requests.append(result.final.t)
                        result.snapshot_times.append(result.final.t)
                        result.snapshots.append(result.final)
                    member_dir = directory/f'{label}-repeat-{repeat}'
                    effective = {**config, 'fixed_step': h, 'strict_snapshots': False,
                                 'error_indicator': 'max_velocity_scalar_defect' if h is None else None}
                    save_member(member_dir, effective, 'sdirk3_mrsav', 'I' if h is None else 'fixed',
                                result, source, cpu_history=cpu_history if h is not None else None)
                    save_stage_checks(member_dir, result)
                    stages = [s for step in result.stages for s in step]
                    member = {'label': label, 'fixed_step': h, 'repeat': repeat,
                              'path': str(member_dir), 'status': result.status, 'error': result.error,
                              'setup_cpu_seconds': setup_cpu, 'integration_cpu_seconds': cpu,
                              'integration_wall_seconds': result.seconds, 'final_time': result.final.t,
                              'accepted_steps': len(result.times)-1,
                              'max_abs_r': max(abs(d['r']) for d in result.diagnostics),
                              'max_stage_residual': max((s['residual'] for s in stages), default=0.),
                              'max_stage_divergence_inf': max((s['divergence_inf'] for s in stages), default=0.)}
                    query_times.extend(result.snapshot_times)
                    record['members'].append(member)
                    write_json(campaign_path, record)
                    log(f'END {label} repeat={repeat} status={result.status} t={result.final.t:.6f} '
                        f'cpu_s={cpu:.2f} error={result.error}')
                    del result, model, initial, stages
            queries = reference_queries(query_times, config)
            record['reference_query_times'] = queries
            write_json(campaign_path, record)
            for i, h in enumerate(spec['reference_steps']):
                reference = run_reference(config, h, directory/f'reference-{i}', source, log,
                                          query_times=queries)
                record['references'].append(reference)
                write_json(campaign_path, record)
                if reference['status'] != 'complete':
                    raise RuntimeError('Shared reference failed')
            record['status'] = ('complete' if all(m['status'] == 'complete' for m in record['members'])
                                else 'complete_with_failures')
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            write_json(campaign_path, record)
            log(f'FINAL status={record["status"]} campaign={campaign_path}')
    return campaign_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    campaign = run(args.config, pilot=args.pilot)
    if args.report:
        from experiments.kolmogorov_adaptive.analyze_adaptive_fixed import analyze
        print(f'REPORT={analyze(campaign)}', flush=True)

"""Add fixed controls to an explicit campaign, retaining verified old runs/references.

Execute against a preserved source root when extending an older campaign. New
steps must visit the existing uniform output times. Never recompute references
or change the parent campaign. Store the entrypoint checksum separately from
the numerical source snapshot; analyze with the current plotting script.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import process_time
from uuid import uuid4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-campaign', type=Path, required=True)
    parser.add_argument('--steps', type=float, nargs='+', required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--analysis-script', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    from experiments.kolmogorov_adaptive.search_tolerance import _restart_single_thread
    _restart_single_thread()

import numpy as np

from experiments.workflow import PROJECT, digest, provenance, write_json
from experiments.kolmogorov_adaptive.run import model_and_initial, output_times, save_member
from experiments.kolmogorov_adaptive.run_adaptive_fixed import save_stage_checks
from solver.integrate import integrate, step_sizes
from solver.mac.kernels import warmup
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV


def extend(parent_path: Path, steps: list[float], *, root: Path = PROJECT) -> Path:
    parent = json.loads(parent_path.read_text())
    config = parent['config']
    if parent['status'] != 'complete' or provenance() != parent['source']:
        raise ValueError('Complete parent and identical numerical source/environment required')
    old_steps = parent['specification']['fixed_steps']
    if not steps or len(set(steps)) != len(steps) or not np.isfinite(steps).all() \
            or min(steps) <= 0 or any(h in old_steps for h in steps):
        raise ValueError('New steps must be positive, finite, unique and absent from the parent')
    if any(not np.isclose(config['output_every']/h, round(config['output_every']/h),
                          atol=1e-10, rtol=0) for h in steps):
        raise ValueError('New steps must divide output_every to reuse saved references')
    queries = np.asarray(parent['reference_query_times'])
    for t in output_times(config):
        if np.min(abs(queries-t)) > 1e-8:
            raise ValueError('Existing reference queries do not cover new outputs')
    for member in parent['members']+parent['references']:
        directory = Path(member['path'])
        manifest = json.loads((directory/'manifest.json').read_text())
        actual = json.loads((directory/'config.json').read_text())
        if manifest['status'] != 'complete' or manifest['source'] != parent['source'] \
                or any(actual.get(k) != v for k, v in config.items() if k != 'reference_step') \
                or digest(directory/'results.npz') != manifest['result_sha256']:
            raise ValueError(f'Invalid reusable member: {directory}')
        if 'stage_checks_sha256' in manifest and \
                digest(directory/'stage_checks.npz') != manifest['stage_checks_sha256']:
            raise ValueError(f'Invalid reusable stage checks: {directory}')
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory = root/'runs/kolmogorov_adaptive/adaptive_fixed'/identity
    directory.mkdir(parents=True)
    campaign = directory/'campaign.json'
    record = {**parent, 'status': 'running', 'parent_campaign': str(parent_path.resolve()),
              'parent_campaign_sha256': digest(parent_path), 'added_fixed_steps': steps,
              'numerical_source_root': str(PROJECT),
              'extension_entrypoint': str(Path(__file__).resolve()),
              'extension_entrypoint_sha256': digest(Path(__file__)),
              'specification': {**parent['specification'],
                                'purpose': 'Extend the verified parent with additional same-scheme fixed controls',
                                'fixed_steps': sorted(old_steps+steps, reverse=True)},
              'members': list(parent['members'])}
    write_json(campaign, record)
    with (directory/'run.log').open('w', buffering=1) as stream:
        def log(message: str) -> None:
            print(message, flush=True)
            stream.write(message+'\n')
        log(f'CAMPAIGN={campaign} parent={parent_path} added_steps={steps}')
        try:
            start = process_time()
            warmup()
            model, initial = model_and_initial(config)
            SDIRK3MRSAV(config['gamma']).step(model, initial, config['initial_step'])
            record['extension_warmup_cpu_seconds'] = process_time()-start
            del model, initial
            for repeat in range(record['specification']['repeats']):
                order = steps if repeat % 2 == 0 else steps[::-1]
                if repeat >= 2:
                    offset = repeat % len(order)
                    order = order[offset:]+order[:offset]
                for h in order:
                    label = f'fixed-{h:g}'
                    setup = process_time()
                    model, initial = model_and_initial(config)
                    setup_cpu = process_time()-setup
                    history: list[float] = []
                    log(f'START {label} repeat={repeat}')
                    start = process_time()
                    def progress(count: int, t: float, elapsed: float) -> None:
                        history.append(process_time()-start)
                        if count == 1 or count % config['progress_every'] == 0:
                            log(f'PROGRESS {label} repeat={repeat} steps={count} '
                                f't={t:.6f}/{config["T"]} integration_wall_s={elapsed:.1f}')
                    result = integrate(model, SDIRK3MRSAV(config['gamma']), initial,
                                       step_sizes(config['T'], dt=h),
                                       snapshots=output_times(config), progress=progress)
                    cpu = process_time()-start
                    target = directory/f'{label}-repeat-{repeat}'
                    effective = {**config, 'fixed_step': h, 'strict_snapshots': False,
                                 'error_indicator': None}
                    save_member(target, effective, 'sdirk3_mrsav', 'fixed', result,
                                record['source'], cpu_history=history)
                    save_stage_checks(target, result)
                    stages = [s for stage in result.stages for s in stage]
                    record['members'].append({
                        'label': label, 'fixed_step': h, 'repeat': repeat, 'path': str(target),
                        'status': result.status, 'error': result.error,
                        'setup_cpu_seconds': setup_cpu, 'integration_cpu_seconds': cpu,
                        'integration_wall_seconds': result.seconds, 'final_time': result.final.t,
                        'accepted_steps': len(result.times)-1,
                        'max_abs_r': max(abs(d['r']) for d in result.diagnostics),
                        'max_stage_residual': max((s['residual'] for s in stages), default=0.),
                        'max_stage_divergence_inf': max((s['divergence_inf'] for s in stages), default=0.)})
                    write_json(campaign, record)
                    log(f'END {label} repeat={repeat} status={result.status} '
                        f't={result.final.t:.6f} cpu_s={cpu:.2f} error={result.error}')
                    del result, stages, model, initial
            record['status'] = ('complete' if all(m['status'] == 'complete' for m in record['members'])
                                else 'complete_with_failures')
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            write_json(campaign, record)
            log(f'FINAL status={record["status"]} campaign={campaign}')
    return campaign


if __name__ == '__main__':
    campaign_path = extend(args.parent_campaign, args.steps, root=args.output_root)
    if args.analysis_script:
        completed = subprocess.run([sys.executable, str(args.analysis_script), '--campaign',
                                    str(campaign_path)], check=True, capture_output=True, text=True)
        print('REPORT='+completed.stdout.strip(), flush=True)

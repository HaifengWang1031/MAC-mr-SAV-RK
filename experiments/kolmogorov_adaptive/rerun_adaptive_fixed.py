"""Run a new adaptive tolerance while preserving the parent comparisons.

Only rtol_velocity may change. Fixed controls are reused with their original
configuration and timing. At new snapshot times, advance each existing SDIRK3
reference from its preceding saved state, recording this checkpoint provenance.
Two reference levels are still required; plotting never initiates computation.
"""
import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
from typing import Any
from uuid import uuid4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-campaign', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--replay-script', type=Path)
    parser.add_argument('--analysis-script', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    from experiments.kolmogorov_adaptive.search_tolerance import _restart_single_thread
    _restart_single_thread()

import numpy as np

from experiments.workflow import PROJECT, digest, provenance, write_json
from experiments.kolmogorov_adaptive.analyze import read_member
from experiments.kolmogorov_adaptive.run import (
    model_and_initial, output_times, reference_schedule, save_member)
from experiments.kolmogorov_adaptive.run_adaptive_fixed import (
    reference_queries, resolved_config, save_stage_checks)
from solver.adaptivity import IController, ScalarErrorControl, integrate_adaptive
from solver.mac.kernels import warmup
from solver.schemes.sdirk3 import SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV


def checked_input(path: Path, source: dict, config: dict) -> dict:
    actual = json.loads((path/'config.json').read_text())
    manifest = json.loads((path/'manifest.json').read_text())
    if (manifest['status'] != 'complete' or manifest['source'] != source
            or any(actual.get(k) != v for k, v in config.items() if k != 'reference_step')
            or digest(path/'results.npz') != manifest['result_sha256']):
        raise ValueError(f'Incompatible parent input: {path}')
    if 'stage_checks_sha256' in manifest and digest(path/'stage_checks.npz') != manifest['stage_checks_sha256']:
        raise ValueError(f'Corrupt parent stage checks: {path}')
    return {'path': str(path.resolve()), 'config_sha256': digest(path/'config.json'),
            'manifest_sha256': digest(path/'manifest.json'), 'result_sha256': manifest['result_sha256'],
            'original_rtol_velocity': config['rtol_velocity']}


def checkpoint_reference(config: dict, parent_config: dict, parent: dict,
                         queries: list[float], directory: Path, source: dict,
                         log: Callable[[str], None]) -> dict:
    """Evaluate new times by forward integration from verified reference states."""
    parent_path = Path(parent['path'])
    identity = checked_input(parent_path, source, parent_config)
    _, original, data = read_member(parent_path)
    nodes = data['output_times']
    step = parent['step']
    if not queries or queries[0] != 0 or np.any(np.diff(queries) <= 0) or queries[-1] > nodes[-1]+1e-10:
        raise ValueError('Reference queries must be ordered and covered by the parent')
    directory.mkdir()
    write_json(directory/'config.json', {**config, 'scheme': 'sdirk3', 'controller': 'fixed',
                                         'reference_step': step})
    manifest: dict[str, Any] = {
        'status': 'running', 'source': source, 'reference': True,
        'reference_policy': 'each query advanced from preceding saved parent SDIRK3 state',
        'parent_reference': identity, 'parent_reference_cpu_seconds': original['integration_cpu_seconds']}
    write_json(directory/'manifest.json', manifest)
    model, _ = model_and_initial(config)
    scheme = SDIRK3()
    saved, checkpoint_indices = [], []
    copied, count = 0, 0
    residual, divergence = 0., 0.
    cpu_start, wall_start = process_time(), perf_counter()
    try:
        for target in queries:
            nearest = int(np.argmin(abs(nodes-target)))
            exact = abs(nodes[nearest]-target) <= 1e-10
            index = nearest if exact else int(np.searchsorted(nodes, target, side='right')-1)
            if index < 0:
                raise ValueError('No reference state precedes query')
            state = model.state(float(nodes[index]), model.grid.pack(data['output_u'][index], data['output_v'][index]),
                                float(data['output_r'][index]))
            if exact:
                copied += 1
            else:
                for h in reference_schedule([state.t, target], step):
                    trial = scheme.step(model, state, h)
                    state = trial.state
                    count += 1
                    residual = max(residual, max(s.residual for s in trial.stages))
                    divergence = max(divergence, max(s.divergence_inf for s in trial.stages))
                    if count == 1 or count % config['reference_progress_every'] == 0:
                        log(f'REFERENCE step={step:g} added_steps={count} query={target:.6f} t={state.t:.6f}')
                if abs(state.t-target) > 1e-10*max(1., abs(target)):
                    raise ValueError('Checkpoint reference time mismatch')
            state = model.state(target, model.vector(state), state.r)
            if not np.isfinite(model.vector(state)).all():
                raise ValueError('Nonfinite checkpoint reference')
            saved.append(state)
            checkpoint_indices.append(index)
        manifest.update(status='complete')
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest.update(integration_cpu_seconds=process_time()-cpu_start,
                        integration_wall_seconds=perf_counter()-wall_start,
                        accepted_steps=count, copied_queries=copied, new_queries=len(saved)-copied,
                        added_stage_residual=residual, added_stage_divergence_inf=divergence,
                        stage_residual=max(residual, original.get('stage_residual', 0.)),
                        stage_divergence_inf=max(divergence, original.get('stage_divergence_inf', 0.)))
        if saved:
            np.savez_compressed(directory/'results.npz', output_times=[s.t for s in saved],
                                output_u=np.stack([s.u for s in saved]), output_v=np.stack([s.v for s in saved]),
                                output_r=[s.r for s in saved], parent_checkpoint_index=checkpoint_indices)
            manifest['result_sha256'] = digest(directory/'results.npz')
        write_json(directory/'manifest.json', manifest)
        log(f'END REFERENCE step={step:g} status={manifest["status"]} added_steps={count} '
            f'cpu_s={manifest["integration_cpu_seconds"]:.2f}')
    return {'path': str(directory.resolve()), 'step': step, **manifest}


def run(parent_path: Path, spec_path: Path, *, root: Path = PROJECT) -> Path:
    parent = json.loads(parent_path.read_text())
    spec, config = resolved_config(spec_path)
    previous = parent['config']
    source = provenance()
    if (parent['status'] != 'complete' or parent['source'] != source
            or {k: v for k, v in config.items() if k != 'rtol_velocity'} !=
               {k: v for k, v in previous.items() if k != 'rtol_velocity'}
            or config['rtol_velocity'] == previous['rtol_velocity']
            or sorted(spec['fixed_steps']) != sorted(parent['specification']['fixed_steps'])
            or spec['reference_steps'] != parent['specification']['reference_steps']):
        raise ValueError('A complete identical-source parent and only changed rtol_velocity are required')
    fixed = [m for m in parent['members'] if m['fixed_step'] is not None]
    identities = [checked_input(Path(m['path']), source, previous) for m in fixed]
    for reference in parent['references']:
        checked_input(Path(reference['path']), source, previous)
    directory = root/'runs/kolmogorov_adaptive/adaptive_fixed'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    campaign_path = directory/'campaign.json'
    record: dict[str, Any] = {
        'status': 'running', 'kind': 'production', 'config': config, 'specification': spec,
        'source': source, 'purpose': __doc__.strip(), 'parent_campaign': str(parent_path.resolve()),
        'parent_campaign_sha256': digest(parent_path), 'parent_config': previous,
        'configuration_file': str(spec_path.resolve()), 'configuration_sha256': digest(spec_path),
        'entrypoint_sha256': digest(Path(__file__)), 'numerical_source_root': str(PROJECT),
        'reused_fixed_inputs': identities, 'members': list(fixed), 'references': [],
        'snapshot_policy': 'nearest natural adaptive nodes; actual times retained',
        'timing_policy': 'three new sequential adaptive repeats; old fixed timing retained; reference/replay excluded'}
    write_json(campaign_path, record)
    with (directory/'run.log').open('w', buffering=1) as stream:
        def log(message: str) -> None:
            print(message, flush=True)
            stream.write(message+'\n')
        log(f'CAMPAIGN={campaign_path} rtol_u={config["rtol_velocity"]:g} reused_fixed={len(fixed)}')
        try:
            start = process_time()
            warmup()
            model, initial = model_and_initial(config)
            SDIRK3MRSAV(config['gamma']).step(model, initial, config['initial_step'])
            SDIRK3().step(model, initial, config['initial_step'])
            record['warmup_cpu_seconds'] = process_time()-start
            del model, initial
            query_times = output_times(config)
            for repeat in range(spec['repeats']):
                setup = process_time()
                model, initial = model_and_initial(config)
                setup_cpu = process_time()-setup
                control = IController(atol=config['atol_velocity'], rtol=config['rtol_velocity'],
                    min_step=config['min_step'], max_step=float('inf') if config['max_step'] is None else config['max_step'],
                    safety=config['safety'], min_factor=config['min_factor'], max_factor=config['max_factor'])
                log(f'START adaptive repeat={repeat}')
                start = process_time()
                def progress(count: int, t: float, elapsed: float) -> None:
                    if count == 1 or count % config['progress_every'] == 0:
                        log(f'PROGRESS adaptive repeat={repeat} steps={count} t={t:.6f}/{config["T"]} wall_s={elapsed:.1f}')
                result = integrate_adaptive(model, SDIRK3MRSAV(config['gamma']), initial,
                    config['T'], config['initial_step'], controller=control,
                    scalar_control=ScalarErrorControl(config['atol_scalar'], config['gamma']),
                    strict_snapshots=False, snapshots=output_times(config), progress=progress)
                cpu = process_time()-start
                valid = [i for i, t in enumerate(result.snapshot_requests) if t <= result.final.t+1e-10]
                result.snapshot_requests = [result.snapshot_requests[i] for i in valid]
                result.snapshot_times = [result.snapshot_times[i] for i in valid]
                result.snapshots = [result.snapshots[i] for i in valid]
                if not result.snapshot_times or abs(result.snapshot_times[-1]-result.final.t) > 1e-10:
                    result.snapshot_requests.append(result.final.t)
                    result.snapshot_times.append(result.final.t)
                    result.snapshots.append(result.final)
                target = directory/f'adaptive-repeat-{repeat}'
                save_member(target, {**config, 'fixed_step': None, 'strict_snapshots': False,
                                    'error_indicator': 'max_velocity_scalar_defect'},
                            'sdirk3_mrsav', 'I', result, source)
                save_stage_checks(target, result)
                stages = [s for step in result.stages for s in step]
                record['members'].append({
                    'label': 'adaptive', 'fixed_step': None, 'repeat': repeat, 'path': str(target.resolve()),
                    'status': result.status, 'error': result.error, 'setup_cpu_seconds': setup_cpu,
                    'integration_cpu_seconds': cpu, 'integration_wall_seconds': result.seconds,
                    'final_time': result.final.t, 'accepted_steps': len(result.times)-1,
                    'max_abs_r': max(abs(d['r']) for d in result.diagnostics),
                    'max_stage_residual': max((s['residual'] for s in stages), default=0.),
                    'max_stage_divergence_inf': max((s['divergence_inf'] for s in stages), default=0.)})
                query_times.extend(result.snapshot_times)
                write_json(campaign_path, record)
                log(f'END adaptive repeat={repeat} status={result.status} cpu_s={cpu:.2f}')
                if result.status != 'complete':
                    raise RuntimeError(f'New adaptive integration failed: {result.error}')
                del result, model, initial, stages
            queries = reference_queries(query_times, config)
            record['reference_query_times'] = queries
            write_json(campaign_path, record)
            for i, reference in enumerate(parent['references']):
                record['references'].append(checkpoint_reference(config, previous, reference, queries,
                    directory/f'reference-{i}', source, log))
                write_json(campaign_path, record)
            if digest(parent_path) != record['parent_campaign_sha256']:
                raise ValueError('Parent campaign changed during execution')
            record['status'] = 'complete'
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            write_json(campaign_path, record)
            log(f'FINAL computation={record["status"]} campaign={campaign_path}')
    return campaign_path


if __name__ == '__main__':
    campaign = run(args.parent_campaign, args.config, root=args.output_root)
    if args.analysis_script is not None:
        record = json.loads(campaign.read_text())
        members = [m for m in record['members'] if m['fixed_step'] is None]
        median = float(np.median([m['integration_cpu_seconds'] for m in members]))
        member = min(members, key=lambda m: abs(m['integration_cpu_seconds']-median))
        command = [sys.executable, str(args.analysis_script), '--campaign', str(campaign),
                   '--output-root', str(args.output_root)]
        with np.load(Path(member['path'])/'results.npz') as data:
            has_rejections = bool(np.any(~data['attempt_accepted']))
        if has_rejections:
            if args.replay_script is None:
                raise ValueError('Replay entrypoint required for rejected trials')
            output = subprocess.check_output([sys.executable, str(args.replay_script), '--member', member['path'],
                '--source-root', str(args.source_root), '--output-root', str(args.output_root)], text=True)
            print(output, end='', flush=True)
            replay_path = output.strip().splitlines()[-1]
            command.extend(['--rejected-replay', replay_path])
        output = subprocess.check_output(command, text=True)
        print(output, end='', flush=True)
        report = output.strip().splitlines()[-1]
        write_json(campaign.parent/'completion.json', {'status': 'complete', 'campaign': str(campaign), 'report': report})
        print(f'REPORT={report}\nFINAL workflow=complete', flush=True)

"""Add prescribed fixed-step controls to an existing adaptive batch."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import process_time
from uuid import uuid4

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.workflow import PROJECT, provenance, write_json
from experiments.kolmogorov_adaptive.run import (
    SCHEMES, SDIRK3, make_scheme, model_and_initial, output_times, reference_schedule,
    save_member,
)
from solver.integrate import integrate

FIXED_STEPS = (0.005, 0.001, 0.0005)


def run_fixed_controls(base_batch_path: Path, *, steps: tuple[float, ...] = FIXED_STEPS,
                       root: Path = PROJECT) -> Path:
    base = json.loads(base_batch_path.read_text())
    if base['status'] != 'complete' or len(base['members']) != 8 or not base['reference']:
        raise ValueError('A complete eight-member adaptive batch is required')
    if len(set(steps)) != len(steps) or any(step <= 0 for step in steps):
        raise ValueError('Fixed steps must be distinct and positive')
    config = base['config']
    outputs = output_times(config)
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    batch_dir = root/'runs/kolmogorov_adaptive/batches'/identity
    batch_dir.mkdir(parents=True)
    batch_path = batch_dir/'batch.json'
    source = provenance()
    batch = {'status': 'running', 'config': config, 'source': source,
             'base_batch': str(base_batch_path.resolve()), 'members': base['members'],
             'fixed_members': [], 'reference': base['reference'],
             'fixed_steps': list(steps),
             'comparison_reference_step': min(config['reference_step'], min(steps)/5)}
    write_json(batch_path, batch)
    with (batch_dir/'run.log').open('w', buffering=1) as log:
        try:
            for scheme_name in SCHEMES:
                for step in steps:
                    message = f'{scheme_name} fixed step={step}'
                    print(message, flush=True)
                    log.write(message+'\n')
                    model, initial = model_and_initial(config)
                    cpu_start = process_time()
                    cpu_history: list[float] = []
                    result = integrate(
                        model, make_scheme(scheme_name, config['gamma']), initial,
                        reference_schedule(outputs, step), snapshots=outputs,
                        progress=lambda count, time, wall: cpu_history.append(
                            process_time()-cpu_start))
                    member = root/'runs/kolmogorov_adaptive'/f'{identity}-{scheme_name}-fixed-{step:g}'
                    save_member(member, {**config, 'fixed_step': step}, scheme_name,
                                'fixed', result, source, cpu_history=cpu_history)
                    batch['fixed_members'].append({
                        'scheme': scheme_name, 'controller': 'fixed', 'step': step,
                        'path': str(member.resolve()), 'status': result.status})
                    write_json(batch_path, batch)
                    message = f'{result.status} steps={len(result.times)-1} path={member}'
                    print(message, flush=True)
                    log.write(message+'\n')
            if config['reference_step'] >= min(steps):
                reference_step = batch['comparison_reference_step']
                model, initial = model_and_initial(config)
                result = integrate(model, SDIRK3(), initial,
                                   reference_schedule(outputs, reference_step), snapshots=outputs)
                member = root/'runs/kolmogorov_adaptive'/f'{identity}-reference'
                save_member(member, {**config, 'reference_step': reference_step},
                            'sdirk3', 'fixed', result, source, reference=True)
                batch['reference'] = str(member.resolve())
                write_json(batch_path, batch)
                message = f'reference step={reference_step} status={result.status} path={member}'
                print(message, flush=True)
                log.write(message+'\n')
            batch['status'] = ('complete' if all(
                item['status'] == 'complete' for item in batch['fixed_members'])
                and result.status == 'complete' else 'complete_with_failures')
        except Exception as exc:
            batch.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            write_json(batch_path, batch)
            log.write(f'FINAL status={batch["status"]} batch={batch_path}\n')
    return batch_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-batch', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(run_fixed_controls(args.base_batch, root=args.root))

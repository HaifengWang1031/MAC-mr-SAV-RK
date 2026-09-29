"""Add prescribed fixed-step controls to an existing adaptive batch."""
import argparse
import hashlib
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
from solver.mac.kernels import warmup

# Coarsest control. IMEX-SDIRK2 treats convection explicitly, so its step is bounded by a
# CFL-like condition. A linear probe on this configuration (max|u|~6, hx=2*pi/128) puts the
# onset near max|u|*tau/hx ~ 1, i.e. tau ~ 0.008: tau=0.01 amplifies grid-scale modes by
# 1.6 per step and overflows, tau=0.005 is marginal (+-1.00), tau=0.0025 is damped (0.86).
# SDIRK3 still survives 0.01, but the ladder has to be common to all four schemes.
FIXED_STEPS = (0.005, 0.0025, 0.001, 0.0005)


def _require_usable_base(base: dict, source: dict) -> None:
    """Require a complete, internally consistent adaptive-only base batch."""
    if len(base['members']) != 8 or not base['reference']:
        raise ValueError('A complete eight-member adaptive batch is required')
    if base.get('fixed_members'):
        raise ValueError('The base batch must not contain fixed controls')
    if any(item['status'] != 'complete' for item in base['members']):
        raise ValueError('All eight adaptive members must be complete')
    source_hash = source['code_sha256']
    if base.get('source', {}).get('code_sha256') != source_hash:
        raise ValueError('The base batch was produced by different source code')
    for item in [*base['members'], {'path': base['reference']}]:
        directory = Path(item['path'])
        actual = json.loads((directory/'config.json').read_text())
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest.get('status') != 'complete':
            raise ValueError('All adaptive members and the reference must be complete')
        if manifest.get('source', {}).get('code_sha256') != source_hash or any(
                actual.get(key) != value for key, value in base['config'].items()):
            raise ValueError(f'Incompatible or stale base record: {directory}')
        checksum = hashlib.sha256((directory/'results.npz').read_bytes()).hexdigest()
        if checksum != manifest.get('result_sha256'):
            raise ValueError(f'Checksum mismatch in base record: {directory}')


def run_fixed_controls(base_batch_path: Path, *, steps: tuple[float, ...] = FIXED_STEPS,
                       root: Path = PROJECT) -> Path:
    base = json.loads(base_batch_path.read_text())
    source = provenance()
    _require_usable_base(base, source)
    if len(set(steps)) != len(steps) or any(step <= 0 for step in steps):
        raise ValueError('Fixed steps must be distinct and positive')
    config = base['config']
    outputs = output_times(config)
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    batch_dir = root/'runs/kolmogorov_adaptive/batches'/identity
    batch_dir.mkdir(parents=True)
    batch_path = batch_dir/'batch.json'
    jit_warmup_seconds = warmup()
    batch = {'status': 'running', 'config': config, 'source': source,
             'base_batch': str(base_batch_path.resolve()), 'members': base['members'],
             'fixed_members': [], 'reference': base['reference'],
             'fixed_steps': list(steps),
             'jit_warmup_seconds': jit_warmup_seconds,
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
                    def fixed_progress(count: int, time: float, elapsed: float) -> None:
                        cpu_history.append(process_time()-cpu_start)
                        if count == 1 or count % config.get('progress_every', 1000) == 0:
                            update = (f'progress {scheme_name} fixed={step:g} '
                                      f'steps={count} t={time:.6g}/{config["T"]:g} '
                                      f'elapsed_s={elapsed:.1f}')
                            print(update, flush=True)
                            log.write(update+'\n')
                    result = integrate(
                        model, make_scheme(scheme_name, config['gamma']), initial,
                        reference_schedule(outputs, step), snapshots=outputs,
                        progress=fixed_progress)
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
                def reference_progress(count: int, time: float, elapsed: float) -> None:
                    if count == 1 or count % config.get('reference_progress_every', 10000) == 0:
                        update = (f'progress reference steps={count} '
                                  f't={time:.6g}/{config["T"]:g} elapsed_s={elapsed:.1f}')
                        print(update, flush=True)
                        log.write(update+'\n')
                result = integrate(model, SDIRK3(), initial,
                                   reference_schedule(outputs, reference_step), snapshots=outputs,
                                   progress=reference_progress)
                member = root/'runs/kolmogorov_adaptive'/f'{identity}-reference'
                save_member(member, {**config, 'reference_step': reference_step},
                            'sdirk3', 'fixed', result, source, reference=True)
                batch['reference'] = str(member.resolve())
                write_json(batch_path, batch)
                message = f'reference step={reference_step} status={result.status} path={member}'
                print(message, flush=True)
                log.write(message+'\n')
            reference_complete = json.loads(
                (Path(batch['reference'])/'manifest.json').read_text())['status'] == 'complete'
            batch['status'] = ('complete' if reference_complete and all(
                item['status'] == 'complete' for item in batch['fixed_members'])
                else 'complete_with_failures')
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

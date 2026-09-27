"""Eight I/PI adaptive runs plus a fixed SDIRK3 reference; no analysis here."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import process_time
from typing import Any
from uuid import uuid4

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.workflow import PROJECT, provenance, write_json
from experiments.kolmogorov_adaptive.model import force, initial_velocity
from solver.adaptivity import IController, PIController, integrate_adaptive
from solver.core import Scheme, State
from solver.integrate import integrate
from solver.mac.grid import MACGrid
from solver.mac.stokes import DirectStokes
from solver.mac.tensor_stokes import TensorStokes
from solver.mac_ns import MACNavierStokes
from solver.schemes.sdirk2 import SDIRK2
from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV
from solver.schemes.sdirk3 import SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV

SCHEMES = {'sdirk2': SDIRK2, 'sdirk2_mrsav': SDIRK2MRSAV,
           'sdirk3': SDIRK3, 'sdirk3_mrsav': SDIRK3MRSAV}


def config_check(config: dict) -> None:
    if 'atol_velocity' not in config or 'rtol_velocity' not in config:
        raise ValueError('Velocity error tolerances are required')
    if config['nx'] < 8 or config['ny'] < 8 or config['T'] <= 0 or config['output_every'] <= 0:
        raise ValueError('Invalid grid, duration or output interval')
    if not np.isclose(config['T']/config['output_every'],
                      round(config['T']/config['output_every']), atol=1e-10, rtol=0):
        raise ValueError('T must be a multiple of output_every')
    if config['min_step'] <= 0 or config['max_step'] < config['min_step'] or \
            not config['min_step'] <= config['initial_step'] <= config['max_step']:
        raise ValueError('Invalid adaptive step bounds')
    if config['reference_step'] <= 0 or config['reference_step'] >= config['max_step']:
        raise ValueError('Reference step must be finer than max_step')
    if config['fixed_step'] <= 0 or config['fixed_step'] > config['max_step']:
        raise ValueError('fixed_step must be positive and no larger than max_step')
    if config.get('progress_every', 1000) <= 0 or \
            config.get('reference_progress_every', 10000) <= 0:
        raise ValueError('Progress intervals must be positive')
    if config.get('stokes_backend', 'direct') not in ('direct', 'tensor'):
        raise ValueError('Unknown Stokes backend')
    krylov_tolerance = config.get('stokes_krylov_tolerance', 1e-12)
    max_iterations = config.get('stokes_max_iterations', 100)
    if not np.isfinite(krylov_tolerance) or krylov_tolerance <= 0 or \
            type(max_iterations) is not int or max_iterations < 1:
        raise ValueError('Invalid tensor Stokes controls')


def model_and_initial(config: dict) -> tuple[MACNavierStokes, State]:
    grid = MACGrid(config['nx'],config['ny'],2*np.pi,2*np.pi)
    model = MACNavierStokes(grid,config['nu'],cache_size=config['cache_size'])
    if config.get('stokes_backend', 'direct') == 'tensor':
        model.backend = TensorStokes(
            model.ops,
            tolerance=config['stokes_tolerance'],
            krylov_tolerance=config.get('stokes_krylov_tolerance', 1e-12),
            max_iterations=config.get('stokes_max_iterations', 100),
        )
    else:
        model.backend = DirectStokes(model.ops,cache_size=config['cache_size'],
                                     tolerance=config['stokes_tolerance'])
    load = force(grid,config['m'])
    model.force = lambda t: load
    velocity = initial_velocity(grid,config['epsilon'],config['initial_modes'])
    return model,model.state(0.,velocity)


def make_scheme(name: str, gamma: float) -> Scheme:
    return SCHEMES[name](gamma) if name.endswith('mrsav') else SCHEMES[name]()


def output_times(config: dict) -> list[float]:
    count = round(config['T']/config['output_every'])
    return [config['output_every']*j for j in range(count+1)]


def reference_schedule(outputs: list[float], nominal: float) -> list[float]:
    steps: list[float] = []
    for left,right in zip(outputs[:-1],outputs[1:]):
        ratio = (right-left)/nominal
        nearest = round(ratio)
        count = max(1,nearest if abs(ratio-nearest) <= 1e-12*max(1.,ratio)
                    else int(np.ceil(ratio)))
        steps.extend([(right-left)/count]*count)
    return steps


def save_member(directory: Path, config: dict, scheme_name: str, controller_name: str,
                result: Any, source: dict, *, reference: bool = False,
                cpu_history: list[float] | None = None) -> None:
    from solver.adaptivity.controller import AdaptiveResult
    directory.mkdir(parents=True)
    actual = {**config,'scheme':scheme_name,'controller':controller_name}
    write_json(directory/'config.json',actual)
    manifest = {'status':'running','source':source,'created_utc':datetime.now(timezone.utc).isoformat(),
                'reference':reference}
    write_json(directory/'manifest.json',manifest)
    adaptive = result if isinstance(result,AdaptiveResult) else None
    snapshots = result.snapshots
    values: dict[str, Any] = {
        'times':np.asarray(result.times),
        'diagnostic_kinetic':np.asarray([d['kinetic'] for d in result.diagnostics]),
        'diagnostic_r':np.asarray([d['r'] for d in result.diagnostics]),
        'output_times':np.asarray(result.snapshot_times),
        'output_u':np.stack([s.u for s in snapshots]) if snapshots else np.empty((0,)),
        'output_v':np.stack([s.v for s in snapshots]) if snapshots else np.empty((0,)),
        'output_r':np.asarray([s.r for s in snapshots]),
    }
    if adaptive is not None:
        attempts = adaptive.attempts
        values.update(attempt_t=np.asarray([a['t'] for a in attempts],dtype=float),
                      attempt_h=np.asarray([a['step'] for a in attempts],dtype=float),
                      attempt_error=np.asarray([a['error'] for a in attempts],dtype=float),
                      attempt_accepted=np.asarray([a['accepted'] for a in attempts],dtype=bool),
                      attempt_cpu=np.asarray([a['cpu_seconds'] for a in attempts],dtype=float))
        (directory/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
    else:
        values['attempt_t'] = np.asarray(result.times[:-1])
        values['attempt_h'] = np.diff(result.times)
        values['attempt_error'] = np.full(len(result.times)-1,np.nan)
        values['attempt_accepted'] = np.ones(len(result.times)-1,dtype=bool)
        values['attempt_cpu'] = (np.asarray(cpu_history) if cpu_history is not None else
                                 np.full(len(result.times)-1,np.nan))
    np.savez_compressed(directory/'results.npz',**values)
    manifest.update(status=result.status,error=result.error,seconds=result.seconds,
                    accepted_steps=len(result.times)-1,
                    rejected_trials=int(np.count_nonzero(~values['attempt_accepted'])),
                    result_sha256=hashlib.sha256((directory/'results.npz').read_bytes()).hexdigest())
    write_json(directory/'manifest.json',manifest)


def run_campaign(config: dict, *, root: Path = PROJECT,
                 reuse_batch: Path | None = None) -> Path:
    config_check(config)
    previous = None if reuse_batch is None else json.loads(reuse_batch.read_text())
    if previous is not None and previous['config'] != config:
        raise ValueError('Reuse batch has a different effective configuration')
    reusable = {}
    if previous is not None:
        for item in previous['members'] + previous['fixed_members']:
            if item['status'] != 'complete':
                continue
            directory = Path(item['path'])
            actual = json.loads((directory/'config.json').read_text())
            manifest = json.loads((directory/'manifest.json').read_text())
            if manifest['status'] != 'complete' or any(
                    actual[key] != value for key, value in config.items()):
                raise ValueError(f'Incompatible reused member: {directory}')
            if hashlib.sha256((directory/'results.npz').read_bytes()).hexdigest() != \
                    manifest['result_sha256']:
                raise ValueError(f'Checksum mismatch in reused member: {directory}')
            reusable[item['scheme'], item['controller']] = item
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    batch_dir = root/'runs/kolmogorov_adaptive/batches'/identity
    batch_dir.mkdir(parents=True)
    batch_path = batch_dir/'batch.json'
    source = provenance()
    batch: dict = {'status':'running','config':config,'source':source,
                   'members':[],'fixed_members':[],'reference':None,
                   'reuse_batch':None if reuse_batch is None else str(reuse_batch.resolve())}
    write_json(batch_path,batch)
    outputs = output_times(config)
    with (batch_dir/'run.log').open('w',buffering=1) as log:
        def message(value: str) -> None:
            print(value,flush=True)
            log.write(value+'\n')
        def progress(label: str, count: int, time: float, elapsed: float,
                     interval: int) -> None:
            if count == 1 or count % interval == 0:
                message(f'progress {label} steps={count} t={time:.6g}/{config["T"]:g} '
                        f'elapsed_s={elapsed:.1f}')
        message(f'batch={batch_path.resolve()}')
        message(f'stokes_backend={config.get("stokes_backend", "direct")}')
        try:
            for scheme_name in SCHEMES:
                for name, cls in [('I',IController),('PI',PIController)]:
                    old = reusable.get((scheme_name, name))
                    if old is not None:
                        batch['members'].append(old)
                        write_json(batch_path,batch)
                        message(f'reused scheme={scheme_name} controller={name} path={old["path"]}')
                        continue
                    model,initial = model_and_initial(config)
                    controller = cls(atol=config['atol_velocity'],rtol=config['rtol_velocity'],
                                     safety=config['safety'],min_step=config['min_step'],
                                     max_step=config['max_step'],estimator_order=2 if scheme_name.startswith('sdirk2') else 3)
                    message(f'start scheme={scheme_name} controller={name}')
                    result = integrate_adaptive(model,make_scheme(scheme_name,config['gamma']),initial,
                                                config['T'],config['initial_step'],controller=controller,
                                                snapshots=outputs,strict_snapshots=True,
                                                progress=lambda count, time, elapsed: progress(
                                                    f'{scheme_name}-{name}', count, time, elapsed,
                                                    config.get('progress_every', 1000)))
                    member = root/'runs/kolmogorov_adaptive'/f'{identity}-{scheme_name}-{name}'
                    save_member(member,config,scheme_name,name,result,source)
                    batch['members'].append({'scheme':scheme_name,'controller':name,'path':str(member.resolve()),
                                             'status':result.status})
                    write_json(batch_path,batch)
                    message(f'{result.status} accepted={len(result.times)-1} rejected={len(result.attempts)-len(result.times)+1} path={member}')
            for scheme_name in SCHEMES:
                old = reusable.get((scheme_name, 'fixed'))
                if old is not None:
                    batch['fixed_members'].append(old)
                    write_json(batch_path,batch)
                    message(f'reused scheme={scheme_name} fixed_step={config["fixed_step"]} '
                            f'path={old["path"]}')
                    continue
                model,initial = model_and_initial(config)
                cpu_start = process_time()
                cpu_history: list[float] = []
                def fixed_progress(count: int, time: float, elapsed: float) -> None:
                    cpu_history.append(process_time()-cpu_start)
                    progress(f'{scheme_name}-fixed', count, time, elapsed,
                             config.get('progress_every', 1000))
                message(f'start scheme={scheme_name} fixed_step={config["fixed_step"]}')
                fixed_result = integrate(model,make_scheme(scheme_name,config['gamma']),initial,
                                         reference_schedule(outputs,config['fixed_step']),snapshots=outputs,
                                         progress=fixed_progress)
                member = root/'runs/kolmogorov_adaptive'/f'{identity}-{scheme_name}-fixed'
                save_member(member,config,scheme_name,'fixed',fixed_result,source,cpu_history=cpu_history)
                batch['fixed_members'].append({'scheme':scheme_name,'controller':'fixed',
                                               'path':str(member.resolve()),'status':fixed_result.status})
                write_json(batch_path,batch)
                message(f'{fixed_result.status} fixed_steps={len(fixed_result.times)-1} path={member}')
            old_reference = None if previous is None else previous['reference']
            if old_reference is not None:
                ref_dir = Path(old_reference)
                ref_config = json.loads((ref_dir/'config.json').read_text())
                ref_manifest = json.loads((ref_dir/'manifest.json').read_text())
                if ref_manifest['status'] == 'complete' and all(
                        ref_config[key] == value for key, value in config.items()) and \
                        hashlib.sha256((ref_dir/'results.npz').read_bytes()).hexdigest() == \
                        ref_manifest['result_sha256']:
                    batch['reference'] = str(ref_dir.resolve())
                    message(f'reused reference path={ref_dir}')
            if batch['reference'] is None:
                model,initial = model_and_initial(config)
                message('start reference scheme=sdirk3')
                reference = integrate(model,SDIRK3(),initial,reference_schedule(outputs,config['reference_step']),
                                      snapshots=outputs,
                                      progress=lambda count, time, elapsed: progress(
                                          'reference', count, time, elapsed,
                                          config.get('reference_progress_every', 10000)))
                ref_dir = root/'runs/kolmogorov_adaptive'/f'{identity}-reference'
                save_member(ref_dir,config,'sdirk3','fixed',reference,source,reference=True)
                batch['reference'] = str(ref_dir.resolve())
                message(f'reference={reference.status} path={ref_dir}')
            reference_complete = json.loads((Path(batch['reference'])/'manifest.json').read_text())['status'] == 'complete'
            batch['status'] = ('complete' if reference_complete and
                               all(m['status']=='complete' for m in batch['members']+batch['fixed_members']) else
                               'complete_with_failures')
            write_json(batch_path,batch)
        except Exception as exc:
            batch.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            write_json(batch_path,batch)
            message(batch['error'])
            raise
        finally:
            message(f'FINAL status={batch["status"]} batch={batch_path}')
    return batch_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=PROJECT)
    parser.add_argument('--reuse-batch',type=Path)
    args = parser.parse_args()
    print(run_campaign(json.loads(args.config.read_text()),root=args.root,
                       reuse_batch=args.reuse_batch))

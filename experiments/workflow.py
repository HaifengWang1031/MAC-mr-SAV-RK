"""Reproducible run/reuse/rerun and read-only analysis entry points."""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from time import perf_counter
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
import numpy as np
import h5py
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup
from solver.mac.stokes import DirectStokes
from solver.mac_ns import MACNavierStokes
from solver.schemes.sdirk2 import SDIRK2
from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV
from solver.integrate import Result, integrate, step_sizes
from .problems import initial_velocity, exact_fields, forcing

PROJECT=Path(__file__).resolve().parents[1]
DEFAULTS={'nx':32,'ny':32,'lx':1.,'ly':1.,'nu':.1,'amplitude':.1,
          'T':.1,'dt':.001,'scheme':'sdirk2_mrsav','gamma':1.,'cache_size':4,'snapshots':[]}

def write_json(path: Path, data: dict) -> None:
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)

def digest(path: Path) -> str:
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): hasher.update(chunk)
    return hasher.hexdigest()

def provenance() -> dict:
    hasher=hashlib.sha256()
    files=[PROJECT/'pyproject.toml',PROJECT/'uv.lock',PROJECT/'.python-version']
    for directory in ('solver','experiments','tools'):
        files.extend(sorted((PROJECT/directory).rglob('*.py')))
    for path in files:
        hasher.update(str(path.relative_to(PROJECT)).encode()); hasher.update(path.read_bytes())
    return {'code_sha256':hasher.hexdigest(),'python':sys.version,'platform':platform.platform(),
            'packages':{name:importlib.metadata.version(name) for name in ('numpy','scipy','numba','h5py','sympy','matplotlib')},
            'threads':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')}}

def effective_config(config: dict) -> dict:
    unknown=set(config)-set(DEFAULTS)-{'experiment','steps','lid_speed'}
    if unknown: raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    cfg={**DEFAULTS,**config}
    if cfg.get('experiment') not in ('stokes_mms','ns_mms','decay','cavity'): raise ValueError('Unknown experiment')
    if cfg['experiment']=='cavity':
        cfg['lid_speed']=config.get('lid_speed',1.)
        if not np.isfinite(cfg['lid_speed']) or cfg['lid_speed']<=0: raise ValueError('Positive finite lid_speed required')
        cfg['boundary']='moving_top_lid_stationary_other_walls'
    elif 'lid_speed' in config: raise ValueError('lid_speed is only supported for cavity')
    if cfg['scheme'] not in ('sdirk2','sdirk2_mrsav'): raise ValueError('Unknown scheme')
    MACGrid(cfg['nx'],cfg['ny'],cfg['lx'],cfg['ly'])
    for key in ('nu','T'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError(f'Invalid {key}')
    if not np.isfinite(cfg['amplitude']) or not np.isfinite(cfg['gamma']) or cfg['gamma']<0:
        raise ValueError('Invalid amplitude or gamma')
    if not isinstance(cfg['cache_size'],int) or cfg['cache_size']<1: raise ValueError('Invalid cache size')
    if 'steps' in config:
        if 'dt' in config: raise ValueError('Use dt or steps, not both')
        cfg.pop('dt')
    schedule=step_sizes(cfg['T'],dt=cfg.get('dt'),steps=cfg.get('steps'))
    if any(not np.isfinite(t) or t<0 or t>cfg['T'] for t in cfg['snapshots']):
        raise ValueError('Invalid snapshot times')
    # Record the actual sequence, including a possible shortened final step.
    cfg['actual_steps']=schedule.tolist()
    return cfg

def save_result(path: Path, result: Result, metrics: dict) -> None:
    temporary=path.with_suffix('.tmp')
    with h5py.File(temporary,'w') as data:
        data.attrs['schema_version']=1
        data.attrs['status']=result.status
        data.attrs['metrics_json']=json.dumps(metrics,allow_nan=False)
        data.create_dataset('times',data=result.times)
        data.create_dataset('dt',data=np.diff(result.times))
        for key in result.diagnostics[0]:
            data.create_dataset('diagnostics/'+key,data=[item[key] for item in result.diagnostics])
        for key,value in [('u',result.final.u),('v',result.final.v),('r',result.final.r),('t',result.final.t)]:
            data.create_dataset('final/'+key,data=value)
        if result.final_stages:
            data.create_dataset('final/stage_pressure',data=np.stack([s.pressure for s in result.final_stages]))
        data.create_dataset('snapshots/requested_times',data=result.snapshot_requests)
        data.create_dataset('snapshots/actual_times',data=result.snapshot_times)
        if result.snapshots:
            data.create_dataset('snapshots/u',data=np.stack([s.u for s in result.snapshots]),compression='gzip')
            data.create_dataset('snapshots/v',data=np.stack([s.v for s in result.snapshots]),compression='gzip')
            data.create_dataset('snapshots/r',data=[s.r for s in result.snapshots])
        candidates=np.full((len(result.stages),2,3),np.nan)
        residuals=candidates.copy()
        counts=np.zeros((len(result.stages),2),dtype=int)
        selected=np.zeros_like(counts,dtype=float)
        for n,stages in enumerate(result.stages):
            for i,stage in enumerate(stages):
                count=stage['root_count']; counts[n,i]=count; selected[n,i]=stage['r']
                candidates[n,i,:count]=stage['candidates']; residuals[n,i,:count]=stage['root_residuals']
        for key,value in [('candidates',candidates),('residuals',residuals),('count',counts),('selected',selected)]:
            data.create_dataset('roots/'+key,data=value)
        for key in ('residual','scalar_residual','divergence_inf'):
            data.create_dataset('stages/'+key,data=np.array([[s[key] for s in stages] for stages in result.stages]).reshape(-1,2))
    temporary.replace(path)

def load_record(directory: Path, *, require_complete: bool = False) -> tuple[dict,dict]:
    """Shared integrity gate for reuse and every analysis reader."""
    manifest=json.loads((directory/'manifest.json').read_text())
    if digest(directory/'results.h5')!=manifest.get('results_sha256') or digest(directory/'config.json')!=manifest.get('config_sha256'):
        raise ValueError(f'Record checksum mismatch: {directory}')
    if manifest.get('status') not in ('complete','failed'):
        raise ValueError('Record is not finalized')
    if require_complete and manifest['status']!='complete': raise ValueError('Complete run required')
    return json.loads((directory/'config.json').read_text()),manifest

def verified(manifest: dict, directory: Path) -> bool:
    try:
        load_record(directory,require_complete=True)
        return True
    except (OSError,KeyError,ValueError): return False


def run_experiment(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    cfg=effective_config(config)
    source=provenance()
    identity=hashlib.sha256(json.dumps({'config':cfg,'source':source},sort_keys=True).encode()).hexdigest()
    parent=Path(root)/'runs'/cfg['experiment']; parent.mkdir(parents=True,exist_ok=True)
    if not rerun:
        for candidate in sorted(parent.glob('*/manifest.json')):
            try: old=json.loads(candidate.read_text())
            except (OSError,ValueError): continue
            if old.get('identity')==identity and verified(old,candidate.parent): return candidate.parent
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+identity[:8]+'-'+uuid4().hex[:8]
    directory=parent/run_id; directory.mkdir()
    manifest={'schema_version':1,'run_id':run_id,'identity':identity,'source':source,'status':'running',
              'created_utc':datetime.now(timezone.utc).isoformat(),'rerun':rerun}
    write_json(directory/'config.json',cfg)
    manifest['config_sha256']=digest(directory/'config.json')
    write_json(directory/'manifest.json',manifest)
    grid=MACGrid(cfg['nx'],cfg['ny'],cfg['lx'],cfg['ly'])
    model=MACNavierStokes(grid,cfg['nu'],cache_size=cfg['cache_size'])
    velocity0=np.zeros(grid.size) if cfg['experiment']=='cavity' else initial_velocity(grid,cfg['amplitude'])
    initial=model.state(0.,velocity0)
    result=Result(initial,[0.],[model.diagnostics(initial)])
    metrics: dict={}
    start=perf_counter()
    with (directory/'run.log').open('w',buffering=1) as log:
        log.write(f'run_id={run_id}\nidentity={identity}\n')
        try:
            metrics['jit_warmup_seconds']=warmup()
            if cfg['experiment']=='stokes_mms':
                solved=model.backend.solve(forcing(grid,cfg['nu'],cfg['amplitude'],0.,False),mass=0.,viscosity=cfg['nu'])
                result.final=model.state(0.,solved.velocity)
                result.diagnostics=[model.diagnostics(result.final)]
                exact,pressure=exact_fields(grid,cfg['nu'],cfg['amplitude'],0.)
                metrics.update(velocity_l2_error=grid.norm(solved.velocity-exact),
                               pressure_l2_error=float(np.sqrt(grid.area*np.sum((solved.pressure-pressure)**2))),
                               stokes_residual=solved.residual)
                from solver.core import Stage
                result.final_stages=[Stage(solved.pressure,solved.residual,solved.divergence_inf)]
            else:
                if cfg['experiment']=='ns_mms':
                    model.force=lambda t: forcing(grid,cfg['nu'],cfg['amplitude'],t)
                if cfg['experiment']=='cavity':
                    from .cavity.model import lid_viscous_load
                    boundary_load=lid_viscous_load(grid,cfg['nu'],cfg['lid_speed'])
                    model.force=lambda t: boundary_load
                scheme=SDIRK2() if cfg['scheme']=='sdirk2' else SDIRK2MRSAV(cfg['gamma'])
                result=integrate(model,scheme,initial,cfg['actual_steps'],snapshots=cfg['snapshots'])
                if cfg['experiment']=='ns_mms':
                    exact,_=exact_fields(grid,cfg['nu'],cfg['amplitude'],result.final.t)
                    metrics['velocity_l2_error']=grid.norm(model.vector(result.final)-exact)
            if cfg['experiment']=='cavity' and result.status=='complete':
                z=model.vector(result.final)
                steady_rhs=model.force(result.final.t)-model.nu*(model.ops.K@z)-model.nonlinear(z)
                projected=model.backend.solve(steady_rhs,mass=1.,viscosity=0.)
                metrics['steady_rhs_l2']=grid.norm(projected.velocity)
                metrics['reynolds']=cfg['lid_speed']*cfg['lx']/cfg['nu']
            metrics['integration_seconds']=result.seconds
            if isinstance(model.backend,DirectStokes):
                metrics.update(factorizations=model.backend.factorizations,
                               factorization_seconds=model.backend.factorization_seconds,
                               linear_solve_seconds=model.backend.solve_seconds)
        except Exception as error:
            result.status='failed'; result.error=f'{type(error).__name__}: {error}'
            log.write(traceback.format_exc())
        metrics['total_seconds']=perf_counter()-start
        save_result(directory/'results.h5',result,metrics)
        manifest.update(status=result.status,error=result.error,accepted_steps=len(result.times)-1,
                        final_time=result.final.t,metrics=metrics,results_sha256=digest(directory/'results.h5'),
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory/'manifest.json',manifest)
        log.write(json.dumps({'status':result.status,'error':result.error,'metrics':metrics})+'\n')
    return directory

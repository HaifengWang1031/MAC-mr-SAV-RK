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
    unknown=set(config)-set(DEFAULTS)-{'experiment','steps'}
    if unknown: raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    cfg={**DEFAULTS,**config}
    if cfg.get('experiment') not in ('stokes_mms','ns_mms','decay'): raise ValueError('Unknown experiment')
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

def verified(manifest: dict, directory: Path) -> bool:
    try:
        return manifest['status']=='complete' and digest(directory/'results.h5')==manifest['results_sha256'] and digest(directory/'config.json')==manifest['config_sha256']
    except (OSError,KeyError): return False

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
    initial=model.state(0.,initial_velocity(grid,cfg['amplitude']))
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
                scheme=SDIRK2() if cfg['scheme']=='sdirk2' else SDIRK2MRSAV(cfg['gamma'])
                result=integrate(model,scheme,initial,cfg['actual_steps'],snapshots=cfg['snapshots'])
                if cfg['experiment']=='ns_mms':
                    exact,_=exact_fields(grid,cfg['nu'],cfg['amplitude'],result.final.t)
                    metrics['velocity_l2_error']=grid.norm(model.vector(result.final)-exact)
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

def analyze_runs(inputs: list[Path], *, root: Path = PROJECT) -> Path:
    """Load explicit complete/failed run records. Never invokes integration."""
    if not inputs: raise ValueError('Supply explicit run directories')
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=Path(root)/'reports'/'diagnostics'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    (report/'figures').mkdir(parents=True); (report/'tables').mkdir()
    record: dict={'status':'running','inputs':[],'source':provenance(),'parameters':{'plots':['kinetic','modified_energy','divergence_inf','r']}}
    write_json(report/'analysis.json',record)
    fig,axes_grid=plt.subplots(2,2,figsize=(11,8))
    axes=axes_grid.ravel()
    rows=[]
    try:
        for directory in map(Path,inputs):
            manifest=json.loads((directory/'manifest.json').read_text())
            checksum=digest(directory/'results.h5')
            if checksum!=manifest.get('results_sha256'): raise ValueError(f'Result checksum mismatch: {directory}')
            cfg=json.loads((directory/'config.json').read_text())
            record['inputs'].append({'run_id':manifest['run_id'],'path':os.path.relpath(directory,root),'results_sha256':checksum})
            with h5py.File(directory/'results.h5','r') as data:
                label=f"{cfg['scheme']} {cfg['nx']}x{cfg['ny']} {manifest['run_id'][-8:]}"
                for ax,key in zip(axes,('kinetic','modified_energy','divergence_inf','r')):
                    ax.plot(data['times'][:],data['diagnostics/'+key][:],label=label)
                    ax.set(xlabel='t',ylabel=key); ax.grid(alpha=.25)
                row={'run_id':manifest['run_id'],'status':manifest['status'],'nx':cfg['nx'],'ny':cfg['ny'],
                     'scheme':cfg['scheme'],'dt_max':max(cfg['actual_steps']),
                     'final_time':float(data['final/t'][()]),'max_divergence':float(np.max(data['diagnostics/divergence_inf'][:])),
                     'max_abs_r':float(np.max(np.abs(data['diagnostics/r'][:]))),**manifest['metrics']}
                rows.append(row)
        for ax in axes: ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(report/'figures/diagnostics.png',dpi=180)
        columns=sorted(set().union(*(row.keys() for row in rows)))
        with (report/'tables/summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader(); writer.writerows(rows)
        record['status']='complete'
    except Exception as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        plt.close(fig); write_json(report/'analysis.json',record)
        (report/'analysis.log').write_text(json.dumps(record,indent=2)+'\n')
    return report

"""Long-time branch scan with streamed evidence, separate from read-only analysis.

Purpose: test two unverified spin normalizations against the reported transition.
Controls: zero velocity, stationary no-slip side-2 square, explicitly configured SDIRK3 or SDIRK3-mr-ccSAV.
Observables: probe curl, energy/enstrophy, angular momentum, residuals and snapshots.
No automatic phenotype claim: spectra and refinement must be checked after completion.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import time
import traceback
from typing import Any
import h5py
import numpy as np
import scipy
from solver.schemes.registry import make_scheme
from .model import make_model, probe_vorticity


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def resolve(config: dict) -> dict:
    c = dict(config)
    if c['backend'] not in ('direct','tensor') or isinstance(c['n'],bool) or c['n']<8:
        raise ValueError('Invalid grid/backend')
    for key in ('dt','T','sample_every','snapshot_every','log_seconds','A0','nu_prime','kappa'):
        if not np.isfinite(c[key]) or c[key]<=0:
            raise ValueError(f'Invalid {key}')
    if not np.isfinite(c['q_spin']):
        raise ValueError('Invalid q_spin')
    for key in ('T','sample_every','snapshot_every'):
        if abs(c[key]/c['dt']-round(c[key]/c['dt']))>1e-8:
            raise ValueError(f'{key} must be a multiple of dt')
    c.setdefault('scheme','sdirk3')
    c.setdefault('gamma',1.)
    c.setdefault('r0',0.)
    if c['scheme'] not in ('sdirk3','sdirk3_mrsav') or not np.isfinite([c['gamma'],c['r0']]).all() or c['gamma']<0:
        raise ValueError('Invalid scheme/gamma/r0')
    c.update(nu=c['A0']/c['nu_prime'],initial='zero',
             domain=[0.,2.,0.,2.],coordinate_shift=[-1.,-1.],
             source_status='angular_wavenumber_image_supported_spin_and_amplitude_unverified',
             force_formula='(-2*A0/kappa*sin(kappa*Y)-q_spin*Y/2,2*A0/kappa*sin(kappa*X)+q_spin*X/2)',
             phase=[0.,0.],noise=0.,symmetry_projection=False,
             resume_supported=False)
    return c


def source_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for folder in ('solver','experiments/molenaar_cavity'):
        for p in sorted((root/folder).rglob('*.py')):
            digest.update(str(p.relative_to(root)).encode())
            digest.update(p.read_bytes())
    for name in ('pyproject.toml','uv.lock','.python-version'):
        digest.update((root/name).read_bytes())
    return digest.hexdigest()


def run_case(config: dict, output: str) -> dict:
    c = resolve(config)
    path = Path(output)
    path.mkdir(parents=True,exist_ok=False)
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path/'run.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(handler)
    manifest = dict(status='running',config=c,code_sha256=source_hash(),
                    start_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
                    numpy=np.__version__,scipy=scipy.__version__,h5py=h5py.__version__,python=platform.python_version(),
                    thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    write_json(path/'config.json',c)
    write_json(path/'manifest.json',manifest)
    logger.info('START %s',json.dumps(c))
    started = time.perf_counter()
    step = 0
    try:
        model = make_model(c)
        scheme = make_scheme(c['scheme'],gamma=c['gamma'])
        state = model.state(0.,np.zeros(model.grid.size),c['r0'])
        # Warm-up on an independent state; production starts again at exact zero.
        scheme.step(model,state,c['dt'])
        started = time.perf_counter()
        sample_stride = round(c['sample_every']/c['dt'])
        snap_stride = round(c['snapshot_every']/c['dt'])
        total = round(c['T']/c['dt'])
        names = ['t','probe_omega','kinetic','enstrophy','angular_momentum','power',
                 'dissipation','divergence_inf','max_stage_residual','max_stage_divergence','cfl','r','max_stage_scalar_residual','max_stage_root_residual']
        residual = divergence = scalar_residual = root_residual = 0.
        stage_records = []
        last_log = started
        with h5py.File(path/'results.h5','w') as h:
            h.attrs.update(schema_version=2,status='running',config_json=json.dumps(c),
                           code_sha256=manifest['code_sha256'],field_layout='staggered_mac_yx',
                           boundary='stationary_no_slip',pressure='incremental_stage_multiplier_mean_zero')
            data = h.create_dataset('diagnostics',(0,len(names)),maxshape=(None,len(names)),chunks=(256,len(names)),dtype='f8')
            data.attrs['columns'] = json.dumps(names)
            stage_data = h.create_dataset('stage_records',(0,4,16),maxshape=(None,4,16),chunks=(64,4,16),dtype='f8',compression='gzip',compression_opts=1)
            stage_data.attrs['columns']=json.dumps(['r','root_count']+[f'candidate_{j}' for j in range(5)]+[f'root_residual_{j}' for j in range(5)]+['scalar_residual','momentum_residual','divergence_inf','continuity_residual'])
            stage_data.attrs['time_rule']='row k corresponds to accepted time (k+1)*dt'
            snap = h.create_group('snapshots')
            sr = snap.create_dataset('r',(0,),maxshape=(None,),chunks=True,dtype='f8')
            st = snap.create_dataset('t',(0,),maxshape=(None,),chunks=True,dtype='f8')
            su = snap.create_dataset('u',(0,*state.u.shape),maxshape=(None,*state.u.shape),chunks=(1,*state.u.shape),compression='gzip',compression_opts=1,dtype='f8')
            sv = snap.create_dataset('v',(0,*state.v.shape),maxshape=(None,*state.v.shape),chunks=(1,*state.v.shape),compression='gzip',compression_opts=1,dtype='f8')
            for step in range(total+1):
                if step:
                    trial = scheme.step(model,state,c['dt'])
                    state = model.state(step*c['dt'],model.vector(trial.state),trial.state.r)
                    if not np.isfinite(state.u).all() or not np.isfinite(state.v).all() or not np.isfinite(state.r):
                        raise FloatingPointError('Nonfinite accepted velocity')
                    residual = max(residual,max(s.residual for s in trial.stages))
                    divergence = max(divergence,max(s.divergence_inf for s in trial.stages))
                    scalar_residual = max(scalar_residual,max(s.scalar_residual for s in trial.stages))
                    root_residual = max(root_residual,max((max(s.root_residuals,default=0.) for s in trial.stages),default=0.))
                    rows = []
                    for stage in trial.stages:
                        if len(stage.candidates)>5 or len(stage.candidates)!=len(stage.root_residuals):
                            raise RuntimeError('Unexpected scalar root storage capacity')
                        rows.append([stage.r,len(stage.candidates)]+list(stage.candidates)+[np.nan]*(5-len(stage.candidates))+list(stage.root_residuals)+[np.nan]*(5-len(stage.root_residuals))+[stage.scalar_residual,stage.residual,stage.divergence_inf,stage.continuity_residual])
                    stage_records.append(rows)
                if step%sample_stride==0 or step==total:
                    d = model.diagnostics(state)
                    values = [state.t,probe_vorticity(model,model.vector(state)),d['kinetic'],
                              .5*d['h1_seminorm_squared'],d['angular_momentum'],d['power'],
                              d['dissipation'],d['divergence_inf'],residual,divergence,
                              c['dt']*(np.max(abs(state.u))/model.grid.hx+np.max(abs(state.v))/model.grid.hy),state.r,scalar_residual,root_residual]
                    if not np.isfinite(values).all():
                        raise FloatingPointError('Nonfinite diagnostic')
                    data.resize(data.shape[0]+1,axis=0)
                    data[-1] = values
                    residual = divergence = scalar_residual = root_residual = 0.
                    if stage_records:
                        begin = stage_data.shape[0]
                        stage_data.resize(begin+len(stage_records),axis=0)
                        stage_data[begin:] = np.asarray(stage_records)
                        stage_records.clear()
                if step%snap_stride==0 or step==total:
                    i = st.shape[0]
                    for ds in (st,su,sv,sr):
                        ds.resize(i+1,axis=0)
                    st[i],su[i],sv[i],sr[i] = state.t,state.u,state.v,state.r
                    h.flush()
                now = time.perf_counter()
                if now-last_log>=c['log_seconds'] or step==total:
                    h.flush()
                    logger.info('PROGRESS step=%d/%d t=%.8g elapsed=%.1fs',step,total,state.t,now-started)
                    last_log = now
            final = h.create_group('final')
            final.create_dataset('u',data=state.u)
            final.create_dataset('v',data=state.v)
            final.create_dataset('stage_pressure',data=trial.stages[-1].pressure)
            final.attrs.update(t=state.t,r=state.r)
            h.attrs['status']='complete'
            h.flush()
        with (path/'results.h5').open('rb') as stream:
            result_hash = hashlib.file_digest(stream,'sha256').hexdigest()
        manifest.update(status='complete',final_t=state.t,steps=step,results_sha256=result_hash)
    except Exception as error:
        manifest.update(status='failed',steps=step,error=f'{type(error).__name__}: {error}')
        logger.error('FAILED\n%s',traceback.format_exc())
        if (path/'results.h5').exists():
            with h5py.File(path/'results.h5','r+') as failed:
                failed.attrs['status']='failed'
    manifest.update(seconds=time.perf_counter()-started,end_utc=datetime.now(timezone.utc).isoformat())
    write_json(path/'manifest.json',manifest)
    logger.info('FINAL %s',json.dumps(manifest))
    handler.close()
    logger.removeHandler(handler)
    return dict(path=str(path),status=manifest['status'],seconds=manifest['seconds'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cases = [resolve({**config['base'],**case}) for case in config['cases']]
    args.output.mkdir(parents=True,exist_ok=False)
    campaign: dict[str,Any] = dict(status='running',members=[],configs=cases,code_sha256=source_hash())
    write_json(args.output/'campaign.json',campaign)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [(i,executor.submit(run_case,c,str(args.output/f'case-{i:02d}'))) for i,c in enumerate(cases)]
        for i,future in futures:
            result = future.result()
            campaign['members'].append(dict(index=i,**result))
            write_json(args.output/'campaign.json',campaign)
            print(json.dumps(dict(index=i,**result)),flush=True)
    campaign['status'] = 'complete' if all(x['status']=='complete' for x in campaign['members']) else 'failed'
    write_json(args.output/'campaign.json',campaign)
    if campaign['status']!='complete':
        raise SystemExit(1)

if __name__=='__main__':
    main()

"""All ranks advance one problem; only rank zero writes persistent records."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import os
from datetime import datetime,timezone
from uuid import uuid4
from time import perf_counter
from typing import Any,Callable
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from solver.mac.grid import MACGrid
from solver.core import State,Stage
from solver.integrate import Result
from solver.mac_parallel.integrate import ParallelNS,ParallelSDIRK2,DistributedStage
from experiments.workflow import PROJECT,effective_config,provenance,write_json,save_result,digest,verified


def on_root(comm: Any, action: Callable) -> Any:
    """Propagate root-side validation/I/O failures before entering collectives."""
    message: tuple[str | None, Any] | None = None
    if comm.rank==0:
        try:message=(None,action())
        except Exception as exc:message=(f'{type(exc).__name__}: {exc}',None)
    error,value=comm.bcast(message,root=0)
    if error:raise RuntimeError(error)
    return value


def run_parallel(config: dict,*,root: Path=PROJECT,rerun: bool=False,comm: Any=MPI.COMM_WORLD) -> Path:
    def prepare() -> dict:
        controls={'linear_tolerance':config.get('linear_tolerance',1e-10),'max_iterations':config.get('max_iterations',300)}
        cfg=effective_config({k:v for k,v in config.items() if k not in controls})
        if cfg['experiment'] not in ('decay','ns_mms','cavity'):raise ValueError('Parallel case must be decay, ns_mms or cavity')
        if comm.size>cfg['ny']:raise ValueError('Ranks must not exceed ny')
        if controls['linear_tolerance']<=0 or not np.isfinite(controls['linear_tolerance']) or controls['max_iterations']<1:
            raise ValueError('Invalid linear solver controls')
        cfg.update(controls);cfg.update(backend='petsc_mpi',mpi_ranks=comm.size)
        source=provenance();source.update(petsc_version=PETSc.Sys.getVersion(),mpi_library=MPI.Get_library_version().strip('\0'),
                                          petsc_options=os.environ.get('PETSC_OPTIONS'))
        identity=hashlib.sha256(json.dumps({'config':cfg,'source':source},sort_keys=True).encode()).hexdigest()
        parent=root/'runs/mac_parallel';parent.mkdir(parents=True,exist_ok=True)
        if not rerun:
            for file in sorted(parent.glob('*/manifest.json')):
                try:old=json.loads(file.read_text())
                except (ValueError,OSError):continue
                if old.get('identity')==identity and verified(old,file.parent):return {'reuse':str(file.parent)}
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+identity[:8]+'-'+uuid4().hex[:8]
        directory=parent/run_id;directory.mkdir()
        write_json(directory/'config.json',cfg)
        manifest={'schema_version':1,'run_id':run_id,'identity':identity,'source':source,'status':'running',
                  'config_sha256':digest(directory/'config.json'),'created_utc':datetime.now(timezone.utc).isoformat()}
        write_json(directory/'manifest.json',manifest)
        (directory/'run.log').write_text(f'one distributed run: {comm.size} MPI ranks\n')
        return {'directory':str(directory),'config':cfg,'manifest':manifest}
    prepared=on_root(comm,prepare)
    if 'reuse' in prepared:return Path(prepared['reuse'])
    directory=Path(prepared['directory']);cfg=prepared['config'];manifest=prepared['manifest']
    grid=MACGrid(cfg['nx'],cfg['ny'],cfg['lx'],cfg['ly'])
    comm.Barrier();start=perf_counter()
    model=ParallelNS(grid,cfg['nu'],comm=comm,tolerance=cfg['linear_tolerance'],max_iterations=cfg['max_iterations'],cache_size=cfg['cache_size'])
    state=model.initial(amplitude=cfg['amplitude'] if cfg['experiment']!='cavity' else 0.)
    if cfg['experiment']=='cavity':
        def force(t: float) -> Any:
            value=model.zero();layout=model.stokes.layout
            if layout.end==grid.ny:
                value.array[layout.local_u-grid.nx+1:layout.local_u]=2*cfg['nu']*cfg['lid_speed']/grid.hy**2
            return value
        model.force=force
    elif cfg['experiment']=='ns_mms':
        # Reuse the analytic expressions, evaluating only locally owned faces.
        from experiments.problems import _expressions
        expressions=_expressions(grid.lx,grid.ly,cfg['nu']);amplitude=cfg['amplitude']
        model.force=lambda t:model.stokes.layout.vector(
            lambda x,y:amplitude*expressions[7](x,y,t)+amplitude**2*expressions[8](x,y,t),
            lambda x,y:amplitude*expressions[9](x,y,t)+amplitude**2*expressions[10](x,y,t))
    stepper=ParallelSDIRK2(sav=cfg['scheme']=='sdirk2_mrsav',gamma=cfg['gamma'])
    times=[0.];diagnostics=[model.diagnostics(state)];stage_records=[]
    nodes=np.r_[0.,np.cumsum(cfg['actual_steps'])]
    requests=cfg['snapshots'];indices=[]
    for t in requests:
        distances=np.abs(nodes-t);tie=8*np.finfo(float).eps*max(float(nodes[-1]),abs(t))
        indices.append(int(np.flatnonzero(distances<=distances.min()+tie)[0]))
    snapshots={};last_stages: list[DistributedStage]=[];status='complete';error=''
    def keep(index: int) -> None:
        if index in indices:
            fields=model.stokes.layout.gather(state.velocity)
            if comm.rank==0:
                assert fields is not None
                snapshots[index]=State(state.t,fields[0],fields[1],state.r)
    keep(0)
    for n,dt in enumerate(cfg['actual_steps']):
        try:
            trial=stepper.step(model,state,dt)
            diagnostic=model.diagnostics(trial.state)
            if not all(np.isfinite(v) for v in diagnostic.values()):raise FloatingPointError('Nonfinite diagnostic')
            state.velocity.destroy();state=trial.state
            for old in last_stages:old.pressure.destroy()
            last_stages=trial.stages
            if comm.rank==0:
                times.append(state.t);diagnostics.append(diagnostic)
                stage_records.append([{'r':s.r,'root_count':len(s.candidates),'candidates':s.candidates,
                                       'root_residuals':s.root_residuals,'residual':s.residual,
                                       'scalar_residual':s.scalar_residual,'divergence_inf':s.divergence_inf} for s in last_stages])
            keep(n+1)
            if (n+1)%100==0:
                def progress() -> None:
                    with (directory/'run.log').open('a') as log:log.write(f'accepted={n+1}, t={state.t:.8g}, elapsed={perf_counter()-start:.3f}s\n')
                on_root(comm,progress)
        except Exception as exc:
            status='failed';error=f'{type(exc).__name__}: {exc}';break
    fields=model.stokes.layout.gather(state.velocity)
    pressures=[model.stokes.layout.gather_pressure(s.pressure) for s in last_stages]
    local_storage=model.stokes.layout.local_n
    metrics={'total_seconds':comm.allreduce(perf_counter()-start,op=MPI.MAX),
             'setup_seconds':comm.allreduce(model.stokes.setup_seconds,op=MPI.MAX),
             'linear_solve_seconds':comm.allreduce(model.stokes.solve_seconds,op=MPI.MAX),
             'velocity_precond_seconds':comm.allreduce(model.stokes.preconditioner_seconds[0],op=MPI.MAX),
             'pressure_precond_seconds':comm.allreduce(model.stokes.preconditioner_seconds[1],op=MPI.MAX),
             'max_iterations':max(model.stokes.iterations,default=0),
             'mean_iterations':float(np.mean(model.stokes.iterations)) if model.stokes.iterations else 0.,
             'max_attempts':max(model.stokes.attempts,default=0),
             'mean_attempts':float(np.mean(model.stokes.attempts)) if model.stokes.attempts else 0.,
             'linear_solves':len(model.stokes.iterations),
             'owned_velocity_dofs_per_rank':comm.allgather(local_storage),
             'global_velocity_dofs':grid.size}
    def finish() -> None:
        assert fields is not None
        final=State(state.t,fields[0],fields[1],state.r)
        result=Result(final,times,diagnostics,stages=stage_records,status=status,error=error,seconds=metrics['total_seconds'])
        for stage,p in zip(last_stages,pressures):result.final_stages.append(Stage(p,stage.residual,stage.divergence_inf,stage.r,stage.candidates,stage.root_residuals,stage.scalar_residual))
        for request,index in zip(requests,indices):
            if index in snapshots:
                result.snapshot_requests.append(request);result.snapshot_times.append(snapshots[index].t);result.snapshots.append(snapshots[index])
        save_result(directory/'results.h5',result,metrics)
        manifest.update(status=status,error=error,accepted_steps=len(times)-1,final_time=state.t,metrics=metrics,
                        finished_utc=datetime.now(timezone.utc).isoformat(),results_sha256=digest(directory/'results.h5'))
        write_json(directory/'manifest.json',manifest)
        with (directory/'run.log').open('a') as log:log.write(json.dumps({'status':status,'error':error,'metrics':metrics})+'\n')
    try:on_root(comm,finish)
    finally:
        state.velocity.destroy()
        for stage in last_stages:stage.pressure.destroy()
        model.close()
    return directory


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=PROJECT);parser.add_argument('--rerun',action='store_true')
    args=parser.parse_args();comm=MPI.COMM_WORLD
    config=on_root(comm,lambda:json.loads(args.config.read_text()))
    path=run_parallel(config,root=args.root,rerun=args.rerun,comm=comm)
    status=on_root(comm,lambda:json.loads((path/'manifest.json').read_text())['status'])
    if comm.rank==0:print(path,flush=True)
    if status!='complete':raise SystemExit(1)

if __name__=='__main__':main()

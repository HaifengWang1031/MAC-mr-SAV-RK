"""Bounded per-step timing comparison of SDIRK2 and SDIRK2-mr-ccSAV across grids."""
import sys
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4
import numpy as np
from solver.mac.grid import MACGrid
from solver.mac.kernels import warmup
from solver.mac.stokes import DirectStokes
from solver.mac_ns import MACNavierStokes
from solver.schemes.sdirk2 import SDIRK2
from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV
from experiments.problems import forcing, initial_velocity
from experiments.workflow import PROJECT, write_json, provenance

DEFAULT_GRIDS=((32,32),(64,64),(128,128),(48,32))

def make_scheme(name: str) -> SDIRK2 | SDIRK2MRSAV:
    if name=='sdirk2': return SDIRK2()
    if name=='sdirk2_mrsav': return SDIRK2MRSAV(1.)
    raise ValueError(f'Unknown scheme {name}')

def measure(name: str, nx: int, ny: int, *, dt: float, steps: int, nu: float, amplitude: float) -> dict:
    """Time one throwaway step (first factorization) plus `steps` steady-dt steps."""
    grid=MACGrid(nx,ny,1.,1.)
    model=MACNavierStokes(grid,nu,cache_size=1)
    model.force=lambda t: forcing(grid,nu,amplitude,t)
    scheme=make_scheme(name)
    backend=model.backend
    if not isinstance(backend,DirectStokes): raise RuntimeError('Timing requires the direct backend')
    state=model.state(0.,initial_velocity(grid,amplitude))
    start=perf_counter()
    state=scheme.step(model,state,dt).state
    cold=perf_counter()-start
    solve_before=backend.solve_seconds
    samples=[]
    divergence=0.
    for _ in range(steps):
        start=perf_counter()
        trial=scheme.step(model,state,dt)
        samples.append(perf_counter()-start)
        state=trial.state
        divergence=max(divergence,max(s.divergence_inf for s in trial.stages))
    solve_seconds=backend.solve_seconds-solve_before
    if backend.factorizations!=1: raise RuntimeError('Per-step sample must reuse one factorization')
    if not np.isfinite(samples).all() or divergence>1e-8*(1+np.max(np.abs(state.u))): raise RuntimeError('Invalid timing sample')
    ordered=sorted(samples)
    return {'scheme':name,'nx':nx,'ny':ny,'unknowns':grid.size,'steps':steps,'dt':dt,
            'cold_step_ms':1e3*cold,'min_step_ms':1e3*ordered[0],'median_step_ms':1e3*float(np.median(ordered)),
            'max_step_ms':1e3*ordered[-1],'linear_solve_ms_per_step':1e3*solve_seconds/steps,
            'other_ms_per_step':1e3*(float(np.median(ordered))-solve_seconds/steps),
            'factorizations':backend.factorizations,'max_divergence':divergence}

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--dt',type=float,default=.01)
    parser.add_argument('--nu',type=float,default=.1)
    parser.add_argument('--amplitude',type=float,default=.1)
    parser.add_argument('--grids',type=int,nargs=2,action='append',metavar=('NX','NY'))
    args=parser.parse_args()
    grids=tuple(map(tuple,args.grids)) if args.grids else DEFAULT_GRIDS
    config={'steps':args.steps,'dt':args.dt,'nu':args.nu,'amplitude':args.amplitude,'grids':grids}
    print(f'jit_warmup_seconds={warmup():.3f}',flush=True)
    # Discarded measurement: absorbs first-call splu/library init so every reported cold step is comparable.
    measure('sdirk2',*min(grids,key=lambda g:g[0]*g[1]),dt=args.dt,steps=1,nu=args.nu,amplitude=args.amplitude)
    records=[]
    for nx,ny in grids:
        for name in ('sdirk2','sdirk2_mrsav'):
            record=measure(name,nx,ny,dt=args.dt,steps=args.steps,nu=args.nu,amplitude=args.amplitude)
            records.append(record)
            print(f"{nx}x{ny} {name} unknowns={record['unknowns']} cold={record['cold_step_ms']:.1f}ms "
                  f"median={record['median_step_ms']:.1f}ms solve={record['linear_solve_ms_per_step']:.1f}ms",flush=True)
    for record in records:
        other=next((r for r in records if r['nx']==record['nx'] and r['ny']==record['ny']
                    and r['scheme']=='sdirk2'),None)
        record['ratio_vs_sdirk2']=record['median_step_ms']/other['median_step_ms'] if other else None
    report=PROJECT/'reports/timings'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    report.mkdir(parents=True)
    write_json(report/'timings.json',{'status':'complete','source':provenance(),'config':config,'records':records})
    columns=['nx','ny','unknowns','scheme','steps','dt','cold_step_ms','median_step_ms','min_step_ms',
             'max_step_ms','linear_solve_ms_per_step','other_ms_per_step','ratio_vs_sdirk2']
    with (report/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    print(f'report={report}')
    print(json.dumps([{k:r[k] for k in ('nx','ny','scheme','median_step_ms','ratio_vs_sdirk2')} for r in records]))

if __name__=='__main__': main()

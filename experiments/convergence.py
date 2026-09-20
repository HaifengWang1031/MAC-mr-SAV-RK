"""Same-grid temporal comparisons with an independently refined reference run."""
from pathlib import Path
import json
import csv
import numpy as np
import h5py
from .workflow import PROJECT,analyze_runs,write_json,digest
from solver.mac.grid import MACGrid

def analyze_temporal(inputs: list[Path], reference: Path, refined: Path, *, root: Path = PROJECT) -> Path:
    def load(path):
        cfg=json.loads((path/'config.json').read_text())
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest['status']!='complete' or digest(path/'results.h5')!=manifest.get('results_sha256'):
            raise ValueError('Temporal comparisons require verified complete runs')
        with h5py.File(path/'results.h5') as data:
            g=MACGrid(cfg['nx'],cfg['ny'],cfg['lx'],cfg['ly'])
            return cfg,g,g.pack(data['final/u'][:],data['final/v'][:])
    if len(inputs)<2: raise ValueError('Supply at least two trial runs')
    cfg,g,ref=load(reference)
    finer,_,ref2=load(refined)
    keys=('experiment','nx','ny','lx','ly','nu','amplitude','T','scheme','gamma')
    def compatible(other):
        if any(other[k]!=cfg[k] for k in keys): raise ValueError('Reference physical/grid/scheme configuration mismatch')
    compatible(finer)
    if max(finer['actual_steps'])>=max(cfg['actual_steps']): raise ValueError('Refined reference must use smaller steps')
    gap=g.norm(ref-ref2)
    rows=[]
    for path in inputs:
        trial,_,value=load(path); compatible(trial)
        if max(trial['actual_steps'])<=max(cfg['actual_steps']): raise ValueError('Trial must be coarser than reference')
        rows.append({'run_id':path.name,'dt':max(trial['actual_steps']),'velocity_l2_error':g.norm(value-ref),
                     'error_to_refined_reference':g.norm(value-ref2)})
    rows.sort(key=lambda x:x['dt'],reverse=True)
    if len({row['dt'] for row in rows})!=len(rows): raise ValueError('Trial maximum steps must be distinct')
    if any(row['velocity_l2_error']<=0 for row in rows): raise ValueError('Zero comparison error: convergence order is undefined')
    for k,row in enumerate(rows):
        row['observed_order']=None if k==0 else float(np.log(rows[k-1]['velocity_l2_error']/row['velocity_l2_error'])/np.log(rows[k-1]['dt']/row['dt']))
    report=analyze_runs(inputs+[reference,refined],root=root)
    with (report/'tables/time_convergence.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    positive=[r['velocity_l2_error'] for r in rows if r['velocity_l2_error']>0]
    ratio=gap/min(positive) if positive else None
    record=json.loads((report/'analysis.json').read_text())
    record['temporal_reference']={'reference':reference.name,'refined':refined.name,'l2_gap':gap,
                                  'gap_over_smallest_error':ratio,'refinement_check_passed':ratio is not None and ratio<.1,
                                  'interpretation':'Refinement is a numerical sensitivity check, not a rigorous reference error bound.'}
    write_json(report/'analysis.json',record)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    ax.loglog([r['dt'] for r in rows],[r['velocity_l2_error'] for r in rows],'o-')
    ax.set(xlabel='maximum time step',ylabel='velocity L2 error to same-grid reference'); ax.grid(True,which='both',alpha=.25)
    fig.tight_layout(); fig.savefig(report/'figures/time_convergence.png',dpi=180); plt.close(fig)
    (report/'analysis.log').write_text(json.dumps(record,indent=2)+'\n')
    return report

"""Analysis of explicit stored runs; no numerical stepping."""
from pathlib import Path
from datetime import datetime,timezone
from uuid import uuid4
import json
import os
import numpy as np
import h5py
from .workflow import PROJECT,provenance,write_json,load_record

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
            cfg,manifest=load_record(directory)
            checksum=manifest['results_sha256']
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

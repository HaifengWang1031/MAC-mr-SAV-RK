"""Plot stored snapshots and the recorded history of one completed run; never computes.

Thin command line over `experiments.plotting`; consumes an explicit run directory, verifies
its record, and writes a report under `reports/<name>/<timestamp>-<uuid>/` with one figure
per requested field, a history figure, a per-snapshot table and an `analysis.json` that
records the input checksum and the code provenance.
"""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
from experiments.plotting import field_figure, history_figure, read_run_fields
from experiments.workflow import PROJECT, digest, provenance, write_json


def report(run: Path, *, root: Path=PROJECT, name: str='forced_ns_speed',
           fields: tuple[str,...]=('speed',)) -> Path:
    cfg, grid, times, available, history = read_run_fields(run)
    unknown=[field for field in fields if field not in available]
    if unknown: raise ValueError(f'Unknown fields {unknown}')
    directory = root/f'reports/{name}'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    (directory/'figures').mkdir(parents=True); (directory/'tables').mkdir()
    for field in fields:
        field_figure(grid,times,available[field],directory/f'figures/{field}.png',field)
    history_figure(history,directory/'figures/history.png')
    speed=available['speed']; vorticity=available['vorticity']
    rows=[]
    for time in times:
        rows.append({'time':time,'max_speed':float(speed[time].max()),'rms_speed':float(np.sqrt(np.mean(speed[time]**2))),
                     'max_vorticity':float(np.abs(vorticity[time]).max()),'mean_vorticity':float(vorticity[time].mean())})
    with (directory/'tables/speed.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(directory/'analysis.json',{'status':'complete','run':str(run.resolve()),
               'run_id':Path(run).name,'results_sha256':digest(run/'results.h5'),
               'config':{key:cfg[key] for key in ('experiment','nx','ny','nu','amplitude','force','initial_condition',
                                                  'scheme','dt','T','snapshots')},
               'snapshots':rows,'figures':[f'figures/{field}.png' for field in fields]+['figures/history.png'],
               'source':provenance()})
    return directory


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=PROJECT); parser.add_argument('--name',default='forced_ns_speed')
    parser.add_argument('--fields',nargs='+',default=['speed'],choices=['speed','vorticity'])
    arguments=parser.parse_args()
    print(report(arguments.run,root=arguments.root,name=arguments.name,fields=tuple(arguments.fields)))

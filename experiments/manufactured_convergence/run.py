"""Run the four-format temporal refinement study with explicit, persistent records."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from experiments.workflow import PROJECT, provenance, run_experiment, write_json
from experiments.manufactured_convergence.analyze import analyze_batch

SCHEMES=('sdirk2','sdirk2_mrsav','sdirk3','sdirk3_mrsav')


def run_campaign(config: dict, *, root: Path=PROJECT) -> Path:
    levels=config['k_levels']
    if not levels or any(type(k) is not int for k in levels) or sorted(set(levels))!=levels:
        raise ValueError('k_levels must be increasing distinct integers')
    if config['base']['experiment']!='manufactured_ns':
        raise ValueError('Expected manufactured_ns base configuration')
    directory=root/'runs/manufactured_ns_convergence/batches'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    batch: dict={'status':'running','config':config,'source':provenance(),'members':[],'report':None}
    path=directory/'batch.json';write_json(path,batch)
    with (directory/'run.log').open('w',buffering=1) as log:
        def message(s: str) -> None:
            print(s,flush=True);log.write(s+'\n')
        message(f'batch={path.resolve()}')
        try:
            for k in levels:
                dt=config['tau_base']*2.**(-k)
                for scheme in SCHEMES:
                    message(f'start k={k} scheme={scheme} dt={dt:.12g}')
                    run=run_experiment({**config['base'],'scheme':scheme,'dt':dt},root=root)
                    status=json.loads((run/'manifest.json').read_text())['status']
                    batch['members'].append({'k':k,'scheme':scheme,'path':str(run.resolve()),'status':status})
                    write_json(path,batch)
                    message(f'{status} {run}')
            report=analyze_batch(path,root=root)
            batch['report']=str(report.resolve())
            batch['status']='complete' if all(m['status']=='complete' for m in batch['members']) else 'complete_with_failed_trials'
        except Exception as exc:
            batch.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            message(batch['error'])
            raise
        finally:
            batch['finished_utc']=datetime.now(timezone.utc).isoformat()
            write_json(path,batch)
            message(f'FINAL status={batch["status"]} batch={path} report={batch["report"]}')
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=PROJECT)
    args=parser.parse_args()
    print(run_campaign(json.loads(args.config.read_text()),root=args.root))

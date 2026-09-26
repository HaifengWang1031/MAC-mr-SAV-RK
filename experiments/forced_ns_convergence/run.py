"""Explicit bounded convergence campaign; persistent member paths and progress log."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
from experiments.workflow import PROJECT, write_json, run_experiment, provenance
from experiments.forced_ns_convergence.analyze import analyze_batch


def validate(config: dict) -> None:
    levels=config['k_levels'];base=config['base']
    schemes=config.get('schemes',['sdirk2','sdirk2_mrsav'])
    if schemes not in (['sdirk2','sdirk2_mrsav'],['sdirk3','sdirk3_mrsav']):
        raise ValueError('schemes must be an SDIRK2 or SDIRK3 comparison pair')
    if not levels or any(type(k) is not int or k<0 for k in levels) or sorted(set(levels))!=levels:
        raise ValueError('k_levels must be increasing distinct nonnegative integers')
    if type(config['reference_k']) is not int or config['reference_k']<=max(levels):raise ValueError('Reference must be finer')
    if not np.isfinite(config['tau_base']) or config['tau_base']<=0:raise ValueError('Invalid base step')
    if not 0<config['sensitivity_threshold']<1:raise ValueError('Invalid reference threshold')
    if type(config['max_reference_refinements']) is not int or config['max_reference_refinements']<0:raise ValueError('Invalid refinement limit')
    if base['experiment'] not in ('forced_ns','trig_ns') or not base['snapshots']:raise ValueError('Forced problem and observation times required')
    for time in [base['T']]+base['snapshots']:
        ratio=time/(config['tau_base']*2.**(-min(levels)))
        if not np.isclose(ratio,round(ratio),rtol=0,atol=1e-9):raise ValueError('Observation times must be exact step nodes')


def run_campaign(config: dict, *, root: Path=PROJECT, rerun: bool=False) -> Path:
    validate(config)
    directory=root/f'runs/{config["base"]["experiment"]}_convergence/batches'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    batch: dict={'status':'running','config':config,'source':provenance(),'trials':[],'references':{},'reports':[]}
    manifest=directory/'batch.json';write_json(manifest,batch)
    with (directory/'run.log').open('w',buffering=1) as log:
        def message(text: str) -> None:
            log.write(text+'\n');print(text,flush=True)
        message(f'batch={manifest.resolve()}')
        def compute(scheme: str,k: int) -> Path:
            message(f'start scheme={scheme} k={k} dt={config["tau_base"]*2.**(-k):.12g}')
            path=run_experiment({**config['base'],'scheme':scheme,'dt':config['tau_base']*2.**(-k)},root=root,rerun=rerun)
            state=json.loads((path/'manifest.json').read_text())['status']
            message(f'{state} {path}')
            return path.resolve()
        try:
            pair=config.get('schemes',['sdirk2','sdirk2_mrsav'])
            for k in config['k_levels']:
                for scheme in pair:
                    path=compute(scheme,k);batch['trials'].append({'k':k,'scheme':scheme,'path':str(path)});write_json(manifest,batch)
            ref_k=config['reference_k'];reference=compute(pair[0],ref_k)
            for attempt in range(config['max_reference_refinements']+1):
                refined=compute(pair[0],ref_k+1)
                batch['references']={'reference':str(reference),'refined':str(refined),'reference_k':ref_k}
                write_json(manifest,batch)
                report=analyze_batch(manifest,root=root);batch['reports'].append(str(report.resolve()))
                passed=json.loads((report/'analysis.json').read_text())['reference_passed']
                message(f'reference_passed={passed} report={report}')
                if passed:
                    batch['status']='complete';break
                reference=refined;ref_k+=1
            else:batch['status']='reference_unresolved'
            batch['failed_trials']=[m['path'] for m in batch['trials'] if json.loads((Path(m['path'])/'manifest.json').read_text())['status']!='complete']
            if batch['status']=='complete' and batch['failed_trials']:batch['status']='complete_with_failed_trials'
        except Exception as exc:
            batch.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            message(batch['error']);raise
        finally:
            batch['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(manifest,batch)
            message(f'FINAL status={batch["status"]} batch={manifest}')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=PROJECT);parser.add_argument('--rerun',action='store_true')
    args=parser.parse_args();print(run_campaign(json.loads(args.config.read_text()),root=args.root,rerun=args.rerun))

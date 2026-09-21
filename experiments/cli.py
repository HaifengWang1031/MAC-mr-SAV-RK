"""Thin shared command-line entry points and explicit batch records."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
from uuid import uuid4
import json
from .analysis import analyze_runs
from .workflow import PROJECT,run_experiment,write_json,provenance

def run_batch(batch: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    base=batch['base']; experiment=base['experiment']
    # Reject unknown base experiment before constructing an output path.
    if experiment not in ('stokes_mms','ns_mms','decay','cavity'): raise ValueError('Unknown experiment')
    directory=root/'runs'/experiment/'batches'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    record: dict={'status':'running','config':batch,'source':provenance(),'members':[]}
    write_json(directory/'batch.json',record)
    try:
        for case in batch['cases']:
            config={**base,**case}
            output=run_experiment(config,root=root,rerun=rerun)
            manifest=json.loads((output/'manifest.json').read_text())
            record['members'].append({'run_id':output.name,'path':str(output.relative_to(root)),'status':manifest['status']})
            write_json(directory/'batch.json',record)
        record['status']='complete' if all(x['status']=='complete' for x in record['members']) else 'failed'
    except Exception as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(directory/'batch.json',record)
        (directory/'batch.log').write_text(json.dumps(record,indent=2)+'\n')
    return directory

def run_main(experiment: str) -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--rerun',action='store_true')
    parser.add_argument('--root',type=Path,default=PROJECT)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if 'cases' in config:
        config['base']={**config['base'],'experiment':experiment}
        directory=run_batch(config,root=args.root,rerun=args.rerun)
        status=json.loads((directory/'batch.json').read_text())['status']
    else:
        directory=run_experiment({**config,'experiment':experiment},root=args.root,rerun=args.rerun)
        status=json.loads((directory/'manifest.json').read_text())['status']
    print(directory)
    if status!='complete': raise SystemExit(1)

def analyze_main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--root',type=Path,default=PROJECT)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--refined-reference',type=Path)
    args=parser.parse_args()
    if args.reference is not None:
        if args.refined_reference is None: parser.error('--reference requires --refined-reference')
        from .convergence import analyze_temporal
        print(analyze_temporal(args.runs,args.reference,args.refined_reference,root=args.root))
    elif args.refined_reference is not None: parser.error('--refined-reference requires --reference')
    else: print(analyze_runs(args.runs,root=args.root))

"""Bounded acceptance computations; writes explicit runs and a validation report."""
import sys
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from experiments.analysis import analyze_runs
from experiments.workflow import PROJECT,run_experiment,write_json,provenance
from experiments.convergence import analyze_temporal

def parallel_tier(root: Path = PROJECT) -> dict:
    """Distributed acceptance tier; explicitly skipped when the mpi extra is unavailable.

    Runs the documented default configuration (outer FGMRES, no PETSC_OPTIONS) on a
    small ladder so a broken distributed path fails acceptance instead of only being
    covered by tests. The tier is bounded: 64^2 on 4 ranks, 256^2 on 2 and 512^2 on 4.
    """
    import importlib.util,shutil,subprocess
    for module in ('petsc4py','mpi4py'):
        if importlib.util.find_spec(module) is None:
            return {'status':'skipped','reason':f'{module} not installed (uv sync --locked --extra mpi)'}
    mpiexec=shutil.which('mpiexec')
    if mpiexec is None:
        return {'status':'skipped','reason':'mpiexec not on PATH'}
    records=[]
    for name,ranks in (('ladder_64',4),('ladder_256',2),('ladder_512',4)):
        config=PROJECT/f'experiments/parallel_ns/configs/{name}.json'
        completed=subprocess.run([mpiexec,'-n',str(ranks),sys.executable,
                                  'experiments/parallel_ns/run.py','--config',str(config),'--root',str(root)],
                                 cwd=PROJECT,capture_output=True,text=True)
        report=completed.stdout.strip().splitlines()
        if completed.returncode!=0:
            detail=(completed.stderr or completed.stdout).strip().splitlines()
            raise RuntimeError(f'Parallel tier failed: {name} on {ranks} ranks: {detail[-1] if detail else completed.returncode}')
        directory=Path(report[-1])
        manifest=json.loads((directory/'manifest.json').read_text())
        metrics=manifest['metrics']
        records.append({'config':name,'ranks':ranks,'path':str(directory.relative_to(root)),
                        'status':manifest['status'],'accepted_steps':manifest['accepted_steps'],
                        'iterations':{'mean':metrics['mean_iterations'],'max':metrics['max_iterations']},
                        'attempts':{'mean':metrics['mean_attempts'],'max':metrics['max_attempts']},
                        'seconds':{'total':metrics['total_seconds'],'setup':metrics['setup_seconds'],
                                   'linear_solve':metrics['linear_solve_seconds']}})
        print('parallel',name,ranks,'ranks',manifest['status'],metrics,flush=True)
        if manifest['status']!='complete': raise RuntimeError(str(directory))
    return {'status':'complete','records':records}

def main() -> None:
    records=[]
    for experiment in ('stokes_mms','ns_mms'):
        for scheme in (('sdirk2',) if experiment=='stokes_mms' else ('sdirk2','sdirk2_mrsav')):
            for nx,ny in ((32,32),(64,64),(128,128),(48,32)):
                config={'experiment':experiment,'scheme':scheme,'nx':nx,'ny':ny,'T':.02,'dt':.0005}
                path=run_experiment(config)
                manifest=json.loads((path/'manifest.json').read_text())
                print(experiment,scheme,nx,ny,manifest['status'],manifest['metrics'],flush=True)
                if manifest['status']!='complete': raise RuntimeError(str(path))
                records.append(path)
    spatial=analyze_runs(records)
    temporal=[]
    for scheme in ('sdirk2','sdirk2_mrsav'):
        paths=[]
        for dt in (.01,.005,.0025,.000625,.0003125):
            path=run_experiment({'experiment':'ns_mms','scheme':scheme,'nx':32,'ny':32,'T':.1,'dt':dt})
            paths.append(path)
        report=analyze_temporal(paths[:3],paths[3],paths[4])
        temporal.append(report)
        print('temporal',scheme,report,flush=True)
    decay=[]
    for scheme in ('sdirk2','sdirk2_mrsav'):
        for dt in (.01,.05,.2):
            path=run_experiment({'experiment':'decay','scheme':scheme,'nx':32,'ny':32,'T':1.,'dt':dt,'nu':.01,'amplitude':.5})
            decay.append(path)
            print('decay',scheme,dt,json.loads((path/'manifest.json').read_text())['status'],flush=True)
    decay_report=analyze_runs(decay)
    record={'source':provenance(),'spatial_report':str(spatial.relative_to(PROJECT)),
            'temporal_reports':[str(p.relative_to(PROJECT)) for p in temporal],
            'decay_report':str(decay_report.relative_to(PROJECT)),
            'spatial_runs':[str(p.relative_to(PROJECT)) for p in records],
            'parallel':parallel_tier()}
    write_json(PROJECT/'docs/validation_results.json',record)
    print('Saved docs/validation_results.json',flush=True)

if __name__=='__main__': main()

"""Bounded acceptance computations; writes explicit runs and a validation report."""
import sys
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from experiments.analysis import analyze_runs
from experiments.workflow import PROJECT,run_experiment,write_json,provenance
from experiments.convergence import analyze_temporal

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
            'spatial_runs':[str(p.relative_to(PROJECT)) for p in records]}
    write_json(PROJECT/'docs/validation_results.json',record)
    print('Saved docs/validation_results.json',flush=True)

if __name__=='__main__': main()

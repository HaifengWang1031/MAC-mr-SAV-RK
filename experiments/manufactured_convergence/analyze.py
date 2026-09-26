"""Read an explicit four-scheme batch, measure error, and compile its LaTeX table."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from uuid import uuid4
import h5py
import numpy as np
from experiments.workflow import PROJECT, digest, load_record, provenance, write_json
from experiments.manufactured_convergence.model import exact
from solver.mac.grid import MACGrid
from solver.mac.operators import MACOperators

SCHEMES=('sdirk2','sdirk2_mrsav','sdirk3','sdirk3_mrsav')
LABELS=('SDIRK2','SDIRK2-mr-ccSAV','SDIRK3','SDIRK3-mr-ccSAV')


def tex_number(x: float) -> str:
    if not np.isfinite(x): return r'\texttt{NaN}'
    return f'${x:.3e}$'


def analyze_batch(batch_path: Path, *, root: Path=PROJECT) -> Path:
    batch=json.loads(batch_path.read_text())
    config=batch['config'];levels=config['k_levels'];members=batch['members']
    expected={(k,s) for k in levels for s in SCHEMES}
    if len(members)!=len(expected) or {(m['k'],m['scheme']) for m in members}!=expected:
        raise ValueError('Incomplete or duplicate comparison members')
    rows=[];source=None
    for m in members:
        path=Path(m['path']);cfg,manifest=load_record(path)
        if source is None:source=manifest['source']['code_sha256']
        if manifest['source']['code_sha256']!=source:raise ValueError('Mixed solver source versions')
        if cfg['scheme']!=m['scheme'] or cfg['experiment']!='manufactured_ns' or cfg['nx']!=config['base']['nx'] \
                or cfg['ny']!=config['base']['ny'] or cfg['T']!=config['base']['T'] \
                or not np.isclose(cfg['dt'],config['tau_base']*2.**(-m['k'])):
            raise ValueError('Run/config mismatch')
        if any(cfg[key]!=config['base'][key] for key in ('nu','amplitude','gamma','lx','ly')):
            raise ValueError('Physical configuration mismatch')
        grid=MACGrid(cfg['nx'],cfg['ny'],cfg['lx'],cfg['ly']);ops=MACOperators(grid)
        l2=h1=float('nan')
        with h5py.File(path/'results.h5') as data:
            if manifest['status']=='complete' and abs(float(data['final/t'][()])-cfg['T'])<1e-9:
                velocity=grid.pack(data['final/u'][:],data['final/v'][:])
                difference=velocity-exact(grid,cfg['T'],cfg['amplitude'])
                l2=grid.norm(difference)
                h1=float(np.sqrt(max(grid.inner(difference,ops.K@difference),0.)))
        rows.append({'k':m['k'],'dt':cfg['dt'],'scheme':m['scheme'],'status':manifest['status'],
                     'accepted_steps':manifest['accepted_steps'],'final_time':manifest['final_time'],
                     'L2':l2,'H1':h1,'L2_rate':float('nan'),'H1_rate':float('nan'),
                     'max_abs_r':None,'run':str(path),'results_sha256':manifest['results_sha256']})
        with h5py.File(path/'results.h5') as data:
            if 'diagnostics/r' in data:
                rows[-1]['max_abs_r']=float(np.max(np.abs(data['diagnostics/r'][:])))
    for row in rows:
        previous=next((p for p in rows if p['scheme']==row['scheme'] and p['k']==row['k']-1),None)
        for norm in ('L2','H1'):
            if previous and np.isfinite(previous[norm]) and np.isfinite(row[norm]) and min(previous[norm],row[norm])>0:
                row[norm+'_rate']=float(np.log2(previous[norm]/row[norm]))
    report=root/'reports/manufactured_ns_convergence'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    (report/'tables').mkdir(parents=True)
    (report/'input_batch.json').write_text(json.dumps(batch,indent=2)+'\n')
    record={'status':'running','batch':str(batch_path.resolve()),'batch_sha256':digest(batch_path),
            'source':provenance(),'inputs':[{'path':r['run'],'sha256':r['results_sha256']} for r in rows],
            'error_definition':'MAC-consistent discrete manufactured velocity, p=0; L2 and K-based H1 seminorm'}
    write_json(report/'analysis.json',record)
    try:
        with (report/'tables/errors.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        lines=[r'\begin{table}[htbp]',r'\centering',
               r'\caption{Temporal errors against the MAC-consistent manufactured velocity on a $128^2$ no-slip grid at $T=2$. Failed runs retain their status and display NaN.}',
               r'\label{tab:manufactured-four-schemes}',r'\setlength{\tabcolsep}{3pt}',
               r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{rr'+''.join('rr' for _ in SCHEMES)+'}',r'\toprule',
               r'$k$ & $\tau_k$ & '+' & '.join(r'\multicolumn{2}{c}{'+label+'}' for label in LABELS)+r'\\',
               r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}',
               r' &  & '+' & '.join(['Error & Rate']*4)+r'\\',r'\midrule']
        for k in levels:
            cells=[str(k),f'${config["tau_base"]*2.**(-k):.6g}$']
            for scheme in SCHEMES:
                row=next(r for r in rows if r['k']==k and r['scheme']==scheme)
                cells += [tex_number(row['L2']),f"{row['L2_rate']:.2f}" if np.isfinite(row['L2_rate']) else '--']
            lines.append(' & '.join(cells)+r'\\')
        lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
        (report/'tables/convergence.tex').write_text('\n'.join(lines)+'\n')
        wrapper=r'''\documentclass{article}
\usepackage{booktabs,graphicx}
\usepackage[margin=1in]{geometry}
\begin{document}
\input{tables/convergence.tex}
\end{document}
'''
        (report/'convergence.tex').write_text(wrapper)
        command=['pdflatex','-interaction=nonstopmode','-halt-on-error','convergence.tex']
        with (report/'analysis.log').open('w') as log:
            completed=subprocess.run(command,cwd=report,stdout=log,stderr=subprocess.STDOUT,check=False,timeout=120)
        if completed.returncode!=0:raise RuntimeError(f'LaTeX compilation failed; see {report}/analysis.log')
        record['status']='complete'
    except Exception as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:write_json(report/'analysis.json',record)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True,type=Path)
    args=parser.parse_args();print(analyze_batch(args.batch))

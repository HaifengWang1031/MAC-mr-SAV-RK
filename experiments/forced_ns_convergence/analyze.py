"""Analyze an explicit batch manifest; never starts numerical computation."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import csv
import json
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import h5py
from experiments.workflow import PROJECT, load_record, write_json, provenance, digest
from solver.mac.grid import MACGrid
from solver.mac.operators import MACOperators


def read_run(path: Path) -> dict:
    cfg, manifest=load_record(path)
    grid=MACGrid(cfg['nx'],cfg['ny'],cfg['lx'],cfg['ly'])
    fields={}
    with h5py.File(path/'results.h5') as f:
        for j,(requested,actual) in enumerate(zip(f['snapshots/requested_times'][:],f['snapshots/actual_times'][:])):
            if abs(actual-requested)>1e-9*max(1.,abs(requested)):
                raise ValueError('Snapshot is not on the requested time node')
            fields[float(requested)]=grid.pack(f['snapshots/u'][j],f['snapshots/v'][j])
        max_r=float(np.max(np.abs(f['diagnostics/r'][:])))
        stage=f['roots/selected'][:]
        max_stage_r=float(np.max(np.abs(stage))) if stage.size else 0.
    return {'config':cfg,'manifest':manifest,'fields':fields,'max_r':max_r,'max_stage_r':max_stage_r,'path':str(path)}


def number(value: float) -> str:
    if not np.isfinite(value):return r'\texttt{NaN}'
    mantissa,power=f'{value:.4e}'.split('e')
    return rf'${mantissa}\times10^{{{int(power)}}}$'


def latex_table(rows: list[dict], times: list[float], norm: str, reference_dt: float) -> str:
    norm_label={'L2':r'$L^2$','H1':r'$H^1$ seminorm'}[norm]
    lines=[r'\begin{table}',r'\centering',
           rf'\caption{{Velocity {norm_label} errors and observed rates for $\tau=0.1\,2^{{-k}}$ with integer $k$; common SDIRK2 reference with $\tau_{{\rm ref}}={reference_dt:.12g}$.}}',
           r'\setlength{\tabcolsep}{4pt}',r'\resizebox{\textwidth}{!}{',
           r'\begin{tabular}{l '+ ' '.join(['cc cc']*len(times))+'}',r'\toprule',
           '& '+' & '.join(rf'\multicolumn{{4}}{{c}}{{$T={t:g}$}}' for t in times)+r'\\',
           ' '.join(rf'\cmidrule(lr){{{2+4*j}-{5+4*j}}}' for j in range(len(times))),
           '& '+' & '.join([r'\multicolumn{2}{c}{SDIRK2} & \multicolumn{2}{c}{SDIRK2-mrSAV}']*len(times))+r'\\',
           '$k$ & '+' & '.join(['Error & Rate & Error & Rate']*len(times))+r'\\',r'\midrule']
    for k in sorted({r['k'] for r in rows}):
        cells=[str(k)]
        for t in times:
            for scheme in ('sdirk2','sdirk2_mrsav'):
                row=next(r for r in rows if r['k']==k and r['time']==t and r['scheme']==scheme)
                rate=row[norm+'_rate']
                cells.extend([number(row[norm]), '--' if not np.isfinite(rate) else f'{rate:.2f}'])
        lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}}',r'\end{table}'])+'\n'


def analyze_batch(batch_path: Path, *, root: Path=PROJECT) -> Path:
    batch=json.loads(batch_path.read_text()); params=batch['config']
    refs=[read_run(Path(batch['references'][key])) for key in ('reference','refined')]
    if any(r['manifest']['status']!='complete' or r['config']['scheme']!='sdirk2' for r in refs):
        raise ValueError('Both reference runs must be completed SDIRK2 runs')
    members={(m['k'],m['scheme']) for m in batch['trials']}
    expected={(k,s) for k in params['k_levels'] for s in ('sdirk2','sdirk2_mrsav')}
    if members!=expected or len(members)!=len(batch['trials']):raise ValueError('Incomplete or duplicate trial members')
    trials=[(m,read_run(Path(m['path']))) for m in batch['trials']]
    cfg=refs[0]['config']; times=cfg['snapshots']
    keys=('experiment','nx','ny','lx','ly','nu','amplitude','gamma','T','snapshots','force','initial_condition','boundary')
    for entry in [refs[1]]+[r for _,r in trials]:
        if any(entry['config'][key]!=cfg[key] for key in keys):raise ValueError('Physical/grid configuration mismatch')
        if entry['manifest']['source']['code_sha256']!=refs[0]['manifest']['source']['code_sha256']:
            raise ValueError('Mixed solver source versions')
    if cfg['experiment'] not in ('forced_ns','trig_ns') or refs[1]['config']['dt']>=cfg['dt']:raise ValueError('Invalid reference ordering')
    grid=MACGrid(cfg['nx'],cfg['ny'],cfg['lx'],cfg['ly']);ops=MACOperators(grid)
    def norms(diff):
        return {'L2':grid.norm(diff),'H1':float(np.sqrt(max(grid.inner(diff,ops.K@diff),0.)))}
    rows=[];gaps={t:norms(refs[0]['fields'][t]-refs[1]['fields'][t]) for t in times}
    for member,entry in trials:
        trial=entry['config'];k=member['k']
        if trial['scheme']!=member['scheme'] or not np.isclose(trial['dt'],params['tau_base']*2.**(-k)) or trial['dt']<=cfg['dt']:
            raise ValueError('Trial membership/step mismatch')
        for t in times:
            values=norms(entry['fields'][t]-refs[0]['fields'][t]) if t in entry['fields'] else {'L2':float('nan'),'H1':float('nan')}
            rows.append({'k':k,'dt':trial['dt'],'scheme':trial['scheme'],'time':t,**values,
                         'status':entry['manifest']['status'],'error':entry['manifest'].get('error',''),
                         'max_abs_r':entry['max_r'],'max_stage_abs_r':entry['max_stage_r'],
                         'seconds':entry['manifest']['metrics']['total_seconds'],'run_id':Path(entry['path']).name})
    for row in rows:
        previous=next((r for r in rows if r['k']==row['k']-1 and r['scheme']==row['scheme'] and r['time']==row['time']),None)
        for norm in ('L2','H1'):
            row[norm+'_rate']=(float(np.log2(previous[norm]/row[norm])) if previous is not None and
                               np.isfinite(previous[norm]) and np.isfinite(row[norm]) and min(previous[norm],row[norm])>0 else float('nan'))
    checks=[]
    for t in times:
        for norm in ('L2','H1'):
            errors=[r[norm] for r in rows if r['time']==t and np.isfinite(r[norm])]
            minimum=min(errors) if errors else 0.
            ratio=gaps[t][norm]/minimum if minimum>0 else None
            checks.append({'time':t,'norm':norm,'reference_gap':gaps[t][norm],'gap_over_min_error':ratio,
                           'passed':ratio is not None and ratio<=params['sensitivity_threshold']})
    report=root/f'reports/{cfg["experiment"]}_convergence'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    (report/'tables').mkdir(parents=True);(report/'figures').mkdir()
    write_json(report/'input_batch.json',batch)
    record={'status':'running','batch_snapshot':'input_batch.json','batch':str(batch_path.resolve()),'batch_sha256':digest(batch_path),'source':provenance(),
            'inputs':[{'path':r['path'],'sha256':r['manifest']['results_sha256']} for r in refs+[r for _,r in trials]],
            'reference_checks':checks,'reference_passed':all(c['passed'] for c in checks)}
    write_json(report/'analysis.json',record)
    try:
        with (report/'tables/errors.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        for norm in ('L2','H1'):
            (report/f'tables/{norm}.tex').write_text(latex_table(rows,times,norm,cfg['dt']))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for norm in ('L2','H1'):
            fig,axes=plt.subplots(1,len(times),figsize=(5*len(times),4),squeeze=False)
            for ax,t in zip(axes[0],times):
                for scheme in ('sdirk2','sdirk2_mrsav'):
                    plot_rows=sorted((r for r in rows if r['time']==t and r['scheme']==scheme and np.isfinite(r[norm]) and r[norm]>0),key=lambda r:r['dt'])
                    if plot_rows:ax.loglog([r['dt'] for r in plot_rows],[r[norm] for r in plot_rows],'o-',label=scheme)
                if plot_rows:
                    steps=np.array([r['dt'] for r in plot_rows]);anchor=plot_rows[0][norm]
                    ax.loglog(steps,anchor*(steps/steps[0])**2,'k--',alpha=.5,label='order 2')
                ax.set(title=f'T={t:g}',xlabel='dt',ylabel=f'velocity {norm} error');ax.legend();ax.grid(True,which='both',alpha=.3)
            fig.tight_layout();fig.savefig(report/f'figures/{norm}.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for scheme in ('sdirk2','sdirk2_mrsav'):
            plot_rows=sorted((r for r in rows if r['time']==times[-1] and r['scheme']==scheme and np.isfinite(r['L2']) and r['L2']>0),key=lambda r:r['dt'])
            if plot_rows:axes[0].loglog([r['seconds'] for r in plot_rows],[r['L2'] for r in plot_rows],'o-',label=scheme)
            if scheme=='sdirk2_mrsav' and plot_rows:
                axes[1].loglog([r['dt'] for r in plot_rows],[r['max_stage_abs_r'] for r in plot_rows],'o-')
        axes[0].set(xlabel='total seconds (including warmup/setup)',ylabel='final velocity L2 error');axes[0].legend()
        axes[1].set(xlabel='dt',ylabel='max stage |r|')
        for ax in axes:ax.grid(True,which='both',alpha=.3)
        fig.tight_layout();fig.savefig(report/'figures/cost_and_r.png',dpi=160);plt.close(fig)
        record['status']='complete'
    except Exception as exc:
        record.update(status='failed',error=str(exc));raise
    finally:
        write_json(report/'analysis.json',record);(report/'analysis.log').write_text(json.dumps(record,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True)
    args=parser.parse_args();print(analyze_batch(args.batch))

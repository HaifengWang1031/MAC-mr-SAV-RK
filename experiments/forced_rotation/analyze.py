"""Long-time statistics of the A/B torque experiment; never starts computation.

Consumes explicit run directories (or one batch record), verifies each with
`experiments.workflow.load_record` and writes a report with the three figure groups the plan
asks for: energy/angular-momentum time curves, field snapshots, and the step-size/grid
comparison of the statistics. Reversal detection is deliberately conservative: a sign change
counts only when the angular momentum leaves a threshold band with the new sign and stays
there for a minimum duration, so jitter about zero is not reported as a reversal.
"""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.plotting import field_figure, history_figure, read_run_fields
from experiments.workflow import PROJECT, digest, load_record, provenance, write_json


def load_run(path: Path) -> dict:
    """Snapshot fields, per-step diagnostics and recorded stage roots of one completed run."""
    cfg, manifest = load_record(path, require_complete=True)
    _, grid, times, fields, history = read_run_fields(path)
    with h5py.File(path/'results.h5') as stored:
        selected = stored['roots/selected'][:] if 'roots/selected' in stored else np.zeros((0,2))
        candidates = stored['roots/candidates'][:] if 'roots/candidates' in stored else np.zeros((0,2,3))
        stage_residual = stored['stages/residual'][:] if 'stages/residual' in stored else np.zeros((0,2))
        stage_divergence = stored['stages/divergence_inf'][:] if 'stages/divergence_inf' in stored else np.zeros((0,2))
    return {'path':path,'config':cfg,'manifest':manifest,'grid':grid,'times':times,'fields':fields,
            'history':history,'max_stage_r':float(np.max(np.abs(selected))) if selected.size else 0.,
            'max_stage_residual':float(np.max(stage_residual)) if stage_residual.size else 0.,
            'max_stage_divergence':float(np.max(stage_divergence)) if stage_divergence.size else 0.,
            'candidates':int(candidates.shape[2]) if candidates.size else 0}


def detect_reversals(times: np.ndarray, values: np.ndarray, *, threshold: float,
                     min_duration: float) -> list[dict]:
    """Sign changes that leave `|value| > threshold` and hold the new sign for `min_duration`.

    The excursion timer starts when the new sign first exceeds the threshold and resets while
    the series stays inside the band, so passing through zero (which every real reversal does)
    is not mistaken for indecision. The reported time is therefore the band exit, which lags
    the true zero crossing by `arcsin(threshold/amplitude)`-worth of phase; report the threshold
    with any reversal time, and read the times as upper bounds on when the sign changed.
    """
    if threshold < 0 or min_duration <= 0: raise ValueError('Invalid threshold or duration')
    events=[]; committed=0.; start=None; sign=0.
    for time,value in zip(np.asarray(times,dtype=float),np.asarray(values,dtype=float)):
        if abs(value)<=threshold:
            start=None; continue
        current=float(np.sign(value))
        if current==committed:
            start=None; continue
        if start is None or current!=sign:
            start,sign=float(time),current
        if time-start>=min_duration:
            # The first commitment out of the zero band establishes the sign; only later
            # changes of an established sign are reversals.
            if committed!=0.:
                events.append({'time':start,'confirmed':float(time),'from':committed,'to':sign})
            committed,start=sign,None
    return events


def residence_times(events: list[dict], times: np.ndarray) -> list[float]:
    """Durations between consecutive confirmed reversals, closed by the end of the record."""
    stamps=[event['time'] for event in events]
    if len(stamps)<2: return []
    return list(np.diff(stamps))+[float(times[-1])-stamps[-1]]


def window(record: dict, start: float) -> tuple[np.ndarray,np.ndarray]:
    """Diagnostics after `start`, the transition window the plan wants excluded from statistics."""
    times=record['history']['time']
    return times[times>=start],np.flatnonzero(times>=start)


def energy_balance_residual(record: dict, index: np.ndarray) -> dict:
    """`dE/dt - (P - D)` of the recorded series: how well the stored scalars close the budget."""
    history=record['history']; times=history['time'][index]
    energy=history['kinetic'][index]; power=history['power'][index]; dissipation=history['dissipation'][index]
    if times.size<3: return {'samples':int(times.size),'relative_residual':float('nan')}
    slope=(energy[1:]-energy[:-1])/(times[1:]-times[:-1])
    budget=0.5*((power[1:]+power[:-1])-(dissipation[1:]+dissipation[:-1]))
    scale=max(float(np.mean(np.abs(budget))),1e-300)
    # The window-integrated balance is the robust statement: (E_end-E_start)/T against the
    # mean (P-D). The pointwise maximum is reported too, as a cheap scheme-consistency probe.
    rate=(float(energy[-1])-float(energy[0]))/(float(times[-1])-float(times[0]))
    mean_budget=float(np.mean(budget))
    return {'samples':int(times.size),'mean_budget':mean_budget,
            'mean_power':float(np.mean(power)),'mean_dissipation':float(np.mean(dissipation)),
            'energy_change_rate':rate,'budget_rate':mean_budget,
            'balance_mismatch':abs(rate-mean_budget)/max(abs(mean_budget),1e-300),
            'relative_residual':float(np.max(np.abs(slope-budget))/scale)}


def summary(record: dict, *, start: float, threshold_fraction: float, min_duration: float) -> dict:
    """Boundedness, budget and rotation statistics of one run after the transition window."""
    times,index=window(record,start)
    history=record['history']
    if index.size<3: raise ValueError('Analysis window is too short')
    energy=history['kinetic'][index]; h1=history['h1_seminorm_squared'][index]
    momentum=history['angular_momentum'][index]; scalar=history['r'][index]
    threshold=threshold_fraction*float(np.sqrt(np.mean(momentum**2)))
    events=detect_reversals(times,momentum,threshold=threshold,min_duration=min_duration)
    residence=residence_times(events,times)
    positive=float(np.mean(momentum>threshold)); negative=float(np.mean(momentum<-threshold))
    return {'run_id':record['path'].name,'window':[float(times[0]),float(times[-1])],
            'samples':int(times.size),'dt':float(record['config']['dt']),'scheme':record['config']['scheme'],
            'forcing_kind':record['config']['forcing_kind'],'grid':[record['config']['nx'],record['config']['ny']],
            'energy_mean':float(np.mean(energy)),'energy_std':float(np.std(energy)),
            'energy_min':float(np.min(energy)),'energy_max':float(np.max(energy)),
            'h1_max':float(np.sqrt(np.max(h1))),'r_max':float(np.max(np.abs(scalar))),
            'stage_r_max':record['max_stage_r'],'stage_residual_max':record['max_stage_residual'],
            'stage_divergence_max':record['max_stage_divergence'],
            'momentum_mean':float(np.mean(momentum)),'momentum_std':float(np.std(momentum)),
            'momentum_min':float(np.min(momentum)),'momentum_max':float(np.max(momentum)),
            'threshold':threshold,'reversals':len(events),
            'reversal_rate':float(len(events)/(times[-1]-times[0])),
            # `None` rather than NaN: the project's run and analysis records forbid NaN.
            'residence_mean':float(np.mean(residence)) if residence else None,
            'residence_max':float(np.max(residence)) if residence else None,
            'fraction_positive':positive,'fraction_negative':negative,
            'events':events,**energy_balance_residual(record,index)}


def statistics_figure(summaries: list[dict], destination: Path) -> None:
    """Angular-momentum histograms and residence-time bars across the compared runs."""
    figure,axes=plt.subplots(1,2,figsize=(11.,4.),constrained_layout=True)
    for entry in summaries:
        label=f'{entry["forcing_kind"]} tau={entry["dt"]:g} {entry["grid"][0]}$^2$'
        axes[0].hist([entry['momentum_mean']],bins=1,alpha=.0)
        axes[1].bar([0],[1],alpha=.0)
    labels=[f'{entry["forcing_kind"]} tau={entry["dt"]:g} {entry["grid"][0]}$^2$' for entry in summaries]
    counts=[entry['reversals'] for entry in summaries]
    axes[0].bar(np.arange(len(counts)),counts)
    axes[0].set_xticks(np.arange(len(counts)),labels,rotation=20,ha='right')
    axes[0].set_ylabel('confirmed reversals')
    means=[entry['residence_mean'] if entry['residence_mean'] is not None else 0. for entry in summaries]
    axes[1].bar(np.arange(len(means)),means)
    axes[1].set_xticks(np.arange(len(means)),labels,rotation=20,ha='right')
    axes[1].set_ylabel('mean residence time')
    figure.suptitle('rotation statistics across step size, grid and scheme',fontsize=10)
    figure.savefig(destination,dpi=160); plt.close(figure)


def curves_figure(records: list[dict], destination: Path, *, quantity: str, label: str,
                  markers: bool=False) -> None:
    """One panel per run with the requested diagnostic; reversal times marked when asked."""
    figure,axes=plt.subplots(len(records),1,figsize=(9.,2.2*len(records)),sharex=True,constrained_layout=True)
    for axis,record in zip(np.atleast_1d(axes),records):
        history=record['history']; times=history['time']
        axis.plot(times,history[quantity],lw=.8)
        axis.set_ylabel(label)
        axis.set_title(f'{record["config"]["forcing_kind"]}, tau={record["config"]["dt"]:g}, '
                       f'{record["config"]["nx"]}$^2$',fontsize=9)
        if markers:
            threshold=.1*float(np.sqrt(np.mean(history['angular_momentum']**2)))
            for event in detect_reversals(times,history['angular_momentum'],threshold=threshold,min_duration=1.):
                axis.axvline(event['time'],color='crimson',lw=.8,ls='--')
    axes[-1].set_xlabel('$t$')
    figure.suptitle(label,fontsize=10)
    figure.savefig(destination,dpi=160); plt.close(figure)


def analyze_runs(paths: list[Path], *, root: Path=PROJECT, start: float=0., threshold_fraction: float=.1,
                 min_duration: float=1., name: str='forced_rotation') -> Path:
    """Report over explicit runs: curves, snapshots, statistics table and reversals table."""
    records=[load_run(path) for path in paths]
    summaries=[summary(record,start=start,threshold_fraction=threshold_fraction,
                       min_duration=min_duration) for record in records]
    directory=root/f'reports/{name}'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    (directory/'figures').mkdir(parents=True); (directory/'tables').mkdir()
    curves_figure(records,directory/'figures/energy.png',quantity='kinetic',label='kinetic energy $E$')
    curves_figure(records,directory/'figures/angular_momentum.png',quantity='angular_momentum',
                  label=r'angular momentum $L$',markers=True)
    history_figure(records[0]['history'],directory/'figures/history_first_run.png',
                   title=f'diagnostics of {records[0]["path"].name}')
    for record in records:
        times=record['times']
        if times: field_figure(record['grid'],times,record['fields']['vorticity'],
                               directory/f'figures/vorticity_{record["path"].name}.png','vorticity')
    statistics_figure(summaries,directory/'figures/statistics.png')
    fields=['run_id','forcing_kind','scheme','grid','dt','window','samples','energy_mean','energy_std',
            'energy_min','energy_max','h1_max','r_max','stage_r_max','stage_residual_max','stage_divergence_max',
            'momentum_mean','momentum_std','momentum_min','momentum_max','threshold','reversals','reversal_rate',
            'residence_mean','residence_max','fraction_positive','fraction_negative','mean_power','mean_dissipation',
            'relative_residual']
    with (directory/'tables/summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for entry in summaries: writer.writerow({key:entry[key] for key in fields})
    with (directory/'tables/reversals.csv').open('w',newline='') as reversals:
        rows_writer=csv.writer(reversals)
        rows_writer.writerow(['run_id','time','confirmed','from','to'])
        for entry in summaries:
            for event in entry['events']:
                rows_writer.writerow([entry['run_id'],event['time'],event['confirmed'],event['from'],event['to']])
    write_json(directory/'analysis.json',{'status':'complete','inputs':[
                   {'path':str(record['path'].resolve()),'run_id':record['path'].name,
                    'results_sha256':digest(record['path']/'results.h5')} for record in records],
               'parameters':{'window_start':start,'threshold_fraction':threshold_fraction,
                             'min_duration':min_duration},
               'summaries':summaries,'source':provenance()})
    return directory


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--root',type=Path,default=PROJECT); parser.add_argument('--start',type=float,default=0.)
    parser.add_argument('--threshold-fraction',type=float,default=.1)
    parser.add_argument('--min-duration',type=float,default=1.)
    parser.add_argument('--name',default='forced_rotation')
    arguments=parser.parse_args()
    print(analyze_runs(arguments.runs,root=arguments.root,start=arguments.start,
                       threshold_fraction=arguments.threshold_fraction,min_duration=arguments.min_duration,
                       name=arguments.name))

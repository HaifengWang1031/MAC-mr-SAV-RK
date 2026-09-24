"""Read stored runs and draw their fields and recorded histories; never starts computation.

Shared by `tools/plot_speed_snapshots.py` and `experiments/forced_rotation/analyze.py` so a
field or history figure has a single definition. Every reader goes through
`experiments.workflow.load_record`, so a corrupted or incomplete record fails here rather
than being plotted.
"""
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .workflow import load_record
from solver.mac.grid import MACGrid, Array

FIELD_CMAPS={'speed':('magma',False),'vorticity':('RdBu_r',True)}

def read_run_fields(path: Path) -> tuple[dict, MACGrid, list[float], dict[str,dict[float,Array]], dict[str,Array]]:
    """Recorded snapshots as cell-centred fields plus the per-step diagnostics."""
    cfg, _ = load_record(path, require_complete=True)
    grid = MACGrid(cfg['nx'],cfg['ny'],cfg['lx'],cfg['ly'])
    times: list[float] = []
    fields: dict[str,dict[float,Array]] = {'speed':{}, 'vorticity':{}}
    with h5py.File(path/'results.h5') as stored:
        for index,(requested,actual) in enumerate(zip(stored['snapshots/requested_times'][:],
                                                      stored['snapshots/actual_times'][:])):
            if abs(float(actual)-float(requested))>1e-9*max(1.,abs(float(requested))):
                raise ValueError('Snapshot is not on the requested time node')
            u = stored['snapshots/u'][index]
            v = stored['snapshots/v'][index]
            centre_u = 0.5*(u[:,:-1]+u[:,1:])
            centre_v = 0.5*(v[:-1,:]+v[1:,:])
            time = float(requested)
            times.append(time)
            fields['speed'][time] = np.hypot(centre_u,centre_v)
            fields['vorticity'][time] = ((v[1:-1,1:]-v[1:-1,:-1])/grid.hx
                                         -(u[1:,1:-1]-u[:-1,1:-1])/grid.hy)
        history = {name:stored['diagnostics/'+name][:] for name in stored['diagnostics']}
        history['time'] = stored['times'][:]
    return cfg, grid, sorted(times), fields, history

def field_figure(grid: MACGrid, times: list[float], values: dict[float,Array], destination: Path,
                 field: str) -> None:
    """One panel per snapshot with a scale shared across panels, so the panels stay comparable."""
    cmap, signed = FIELD_CMAPS[field]
    if signed:
        limit=float(max(np.abs(values[t]).max() for t in times)); vmin,vmax=-limit,limit
    else:
        limit=float(max(values[t].max() for t in times)); vmin,vmax=0.,limit
    figure, axes = plt.subplots(1,len(times),figsize=(4.4*len(times),4.2),constrained_layout=True)
    for axis,time in zip(np.atleast_1d(axes),times):
        snapshot=values[time]
        image = axis.imshow(snapshot,origin='lower',extent=[0,grid.lx,0,grid.ly],cmap=cmap,
                            vmin=vmin,vmax=vmax,interpolation='nearest')
        if field=='vorticity':
            label, moment = 'max$|\\omega|$', rf'$\omega_{{\rm rms}}={np.sqrt(np.mean(snapshot**2)):.3f}$'
        else:
            label, moment = 'max$|u|$', rf'$u_{{\rm rms}}={np.sqrt(np.mean(snapshot**2)):.3f}$'
        axis.set_title(rf'$T={time:g}$,  {label}$={np.abs(snapshot).max():.3f}$,  {moment}')
        axis.set_xlabel('$x$'); axis.set_ylabel('$y$'); axis.set_aspect('equal')
        figure.colorbar(image,ax=axis,fraction=.046,pad=.04)
    title=(r'vorticity $\partial_x v-\partial_y u$' if field=='vorticity'
           else r'velocity magnitude $\sqrt{u^2+v^2}$')
    figure.suptitle(rf'{title} of run {grid.nx}$\times${grid.ny}, shared scale',fontsize=10)
    figure.savefig(destination,dpi=160)
    plt.close(figure)

def history_figure(history: dict[str,Array], destination: Path, *, title: str='recorded diagnostics') -> None:
    """Per-step diagnostics on a shared time axis, to show whether the flow is steady."""
    time=history['time']
    candidates=[('kinetic','kinetic energy $E$'),('h1_seminorm_squared','$H^1$ seminorm$^2$'),
                ('dissipation',r'dissipation $\nu\|\nabla u\|^2$'),('power',r'power $(f,u)$'),
                ('angular_momentum',r'angular momentum $L$'),('divergence_inf','max$|Du|$'),
                ('r','SAV $r$')]
    panels=[(name,label) for name,label in candidates if name in history]
    figure, axes = plt.subplots(len(panels),1,figsize=(7.2,1.7*len(panels)),sharex=True,constrained_layout=True)
    for axis,(name,label) in zip(np.atleast_1d(axes),panels):
        axis.plot(time,history[name],lw=1.)
        axis.set_ylabel(label)
        if name=='divergence_inf': axis.set_yscale('symlog',linthresh=1e-14)
    axes[-1].set_xlabel('$t$')
    figure.suptitle(title,fontsize=10)
    figure.savefig(destination,dpi=160)
    plt.close(figure)

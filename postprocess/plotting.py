"""Shared field/history figures and TeX style, separate from record reading and stepping.

Task-specific convergence figures and physical interpretation stay with research.
"""
from pathlib import Path
import glob
import os
import shutil
import contextlib
from collections.abc import Iterator
from typing import Any, cast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.mac.grid import MACGrid, Array

FIELD_CMAPS={'speed':('magma',False),'vorticity':('RdBu_r',True)}

# MacTeX links its binaries at /Library/TeX/texbin (and TeX Live keeps them under
# /usr/local/texlive/<year>/bin/<arch>). Neither is on the PATH of a non-login shell,
# so shutil.which alone reports a false "tool missing" on a machine that has TeX.
LATEX_SEARCH_PATHS=('/Library/TeX/texbin','/usr/local/texlive/*/bin/*','/opt/homebrew/bin','/usr/local/bin')

def latex_binary(name: str='pdflatex') -> str | None:
    """Absolute path of a TeX binary from PATH or the standard TeX Live/MacTeX roots."""
    found = shutil.which(name)
    if found:
        return found
    for pattern in LATEX_SEARCH_PATHS:
        for directory in sorted(glob.glob(pattern),reverse=True):
            candidate = Path(directory)/name
            if candidate.is_file() and os.access(candidate,os.X_OK):
                return str(candidate)
    return None

def usetex_available() -> bool:
    """True when matplotlib can typeset labels with real LaTeX (latex + dvipng or ghostscript)."""
    return (latex_binary('latex') is not None
            and (latex_binary('dvipng') is not None or shutil.which('gs') is not None))

@contextlib.contextmanager
def latex_style(rc: dict[str,Any] | None = None) -> Iterator[None]:
    """Figure style with LaTeX-typeset math, falling back to Computer-Modern mathtext.

    matplotlib shells out to `latex`/`dvipng` by name, so the TeX binary directory is
    prepended to PATH for the duration of the context. Without a toolchain the same
    labels still render through mathtext, so a figure is always produced.
    """
    settings: dict[str,Any] = {'font.family':'serif', 'axes.unicode_minus':False,
                               'axes.labelsize':10, 'axes.titlesize':10,
                               'legend.fontsize':8, 'xtick.labelsize':9, 'ytick.labelsize':9}
    latex = latex_binary('latex') if usetex_available() else None
    if latex is not None:
        settings.update({'text.usetex':True, 'font.size':9,
                         'text.latex.preamble':r'\usepackage{amsmath}'})
    else:
        settings.update({'text.usetex':False, 'mathtext.fontset':'cm'})
    settings.update(rc or {})
    params = cast(Any, plt.rcParams)
    previous_path = os.environ.get('PATH','')
    previous_values = {key:params[key] for key in settings}
    if latex is not None:
        os.environ['PATH'] = str(Path(latex).parent)+os.pathsep+previous_path
    try:
        params.update(settings)
        yield
    finally:
        params.update(previous_values)
        os.environ['PATH'] = previous_path

@contextlib.contextmanager
def _figure_style() -> Iterator[None]:
    """Keep normal caller styles; make an inherited TeX request availability-aware."""
    if plt.rcParams['text.usetex']:
        with latex_style():
            yield
    else:
        yield


@_figure_style()
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

@_figure_style()
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
    np.atleast_1d(axes)[-1].set_xlabel('$t$')
    figure.suptitle(title,fontsize=10)
    figure.savefig(destination,dpi=160)
    plt.close(figure)

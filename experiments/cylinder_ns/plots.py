"""Render only saved coefficients/history; no solver or HDF5 file ownership."""
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy.spatial import cKDTree

from .reconstruct import decode_velocity
from .benchmark import REFERENCE


def plot_mesh(mesh: Any, directory: Path) -> None:
    triangle = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    for name, bounds, size in (('mesh', (0., 2.2, 0., .41), (12., 3.)),
                                ('mesh_zoom', (.08, .6, .04, .37), (7., 4.5))):
        fig, ax = plt.subplots(figsize=size, layout='constrained')
        ax.triplot(triangle, color='#304860', linewidth=.35)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='x', ylabel='y',
               title=f'Saved polygonal cylinder mesh: {mesh.t.shape[1]} triangles')
        ax.set_aspect('equal')
        fig.savefig(directory/(name+'.png'), dpi=160)
        plt.close(fig)


def _display(basis: Any) -> tuple[Any, Any, np.ndarray]:
    mesh = basis.mesh
    display = mesh.refined()
    component = basis.split_indices()[0]
    distance, indices = cKDTree(basis.doflocs[:, component].T).query(display.p.T)
    if np.max(distance) > 1e-12:
        raise ValueError('Display subdivision does not match saved P2 nodes')
    return (mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T),
            mtri.Triangulation(display.p[0], display.p[1], display.t.T), indices)


def _vorticity(basis: Any, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    gradient = basis.interpolate(decode_velocity(basis, u, v)).grad
    return np.sum((gradient[1, 0]-gradient[0, 1])*basis.dx, axis=1)/np.sum(basis.dx, axis=1)


def plot_fields(basis: Any, u: np.ndarray, v: np.ndarray, title: str, directory: Path) -> dict:
    triangle, display, indices = _display(basis)
    speed = np.hypot(u[indices], v[indices])
    vorticity = _vorticity(basis, u, v)
    artist: Any
    for name, values, unit in (('velocity', speed, 'speed'),
                                ('vorticity', vorticity, 'cell-average vorticity (saturation at ±20)')):
        fig, ax = plt.subplots(figsize=(12., 3.), layout='constrained')
        if name == 'velocity':
            artist = ax.tripcolor(display, values, shading='gouraud', cmap='viridis', vmin=0.)
        else:
            artist = ax.tripcolor(triangle, facecolors=values, cmap='RdBu_r', vmin=-20., vmax=20.)
        ax.set(xlim=(0., 2.2), ylim=(0., .41), xlabel='x', ylabel='y', title=title)
        ax.set_aspect('equal')
        fig.colorbar(artist, ax=ax, label=unit, fraction=.025, pad=.02)
        fig.savefig(directory/(name+'.png'), dpi=160)
        plt.close(fig)
    return {'nodal_max_speed': float(np.max(speed)),
            'cell_average_vorticity_range': [float(np.min(vorticity)), float(np.max(vorticity))],
            'velocity_rendering': 'P2 nodal values on a linear four-subtriangle display',
            'vorticity_rendering': 'exact cell averages for affine P2 triangles; color clipped at ±20'}


def plot_histories(times: np.ndarray, steps: np.ndarray, diagnostics: dict, observations: dict,
                   attempts: dict, tolerance: float, directory: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11., 7.), sharex=True, layout='constrained')
    for ax, key, label in zip(axes, ('drag', 'lift', 'pressure_difference'),
                              ('$C_D$', '$C_L$', '$p_{front}-p_{rear}$')):
        ax.plot(observations['time'], observations[key], linewidth=1.)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('time')
    axes[0].set_title('Saved instantaneous physical-pressure/gradient-stress measurements')
    fig.savefig(directory/'forces.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11., 6.), layout='constrained')
    axes[0, 0].semilogy(times[1:], steps)
    axes[0, 0].set_ylabel('accepted step')
    axes[0, 1].plot(times, diagnostics['r'])
    axes[0, 1].set_ylabel('SAV scalar r')
    axes[1, 0].semilogy(times, np.maximum(diagnostics['continuity_residual'], 1e-17))
    axes[1, 0].axhline(tolerance, color='k', linestyle='--', label='solve tolerance')
    axes[1, 0].set_ylabel('complete weak continuity (display floor $10^{-17}$)')
    axes[1, 0].legend(fontsize=8)
    if attempts:
        for key, label in (('error', 'combined'), ('error_u', 'velocity'), ('error_r', 'scalar')):
            values = attempts[key]
            mask = np.isfinite(values)
            if np.any(mask):
                axes[1, 1].semilogy(attempts['t'][mask], np.maximum(values[mask], 1e-16), label=label)
        mask = ~attempts['accepted'] & np.isfinite(attempts['error'])
        axes[1, 1].semilogy(attempts['t'][mask], np.maximum(attempts['error'][mask], 1e-16),
                            'rx', label='rejected')
        axes[1, 1].axhline(1., color='k', linestyle='--')
        axes[1, 1].set_ylabel('normalized embedded/defect errors')
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].plot(times, diagnostics['divergence_inf'])
        axes[1, 1].set_ylabel('quadrature physical div∞ (not weak continuity)')
    for ax in axes.flat:
        ax.set_xlabel('time')
        ax.grid(alpha=.25)
    fig.savefig(directory/'steps.png', dpi=160)
    plt.close(fig)


def plot_cycles(observations: dict, cycles: dict, directory: Path) -> None:
    if 'window' not in cycles:
        return
    start, end = cycles['window']
    time = observations['time']
    selected = (time >= start) & (time <= end)
    fig, axes = plt.subplots(3, 1, figsize=(11., 7.), sharex=True, layout='constrained')
    for ax, name, key, label in zip(axes, ('drag', 'lift', 'pressure'),
                                    ('drag', 'lift', 'pressure_difference'), ('$C_D$', '$C_L$', '$Δp$')):
        ax.plot(time[selected], observations[key][selected], label='saved physical measurements')
        for bound in ('min', 'max'):
            ax.axhline(REFERENCE[name+'_'+bound], color='#b05b22', linestyle='--',
                       label='published sampled envelope' if bound == 'min' else None)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0].set_title('Final three complete lift cycles (reference range, not a certified tolerance band)')
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('time')
    fig.savefig(directory/'force_cycles.png', dpi=160)
    plt.close(fig)


def animate(basis: Any, u: np.ndarray, v: np.ndarray, times: np.ndarray, directory: Path) -> dict | None:
    actual, indices = np.unique(times, return_index=True)
    if len(actual) < 3:
        return None
    triangle, display, nodes = _display(basis)
    speed_max = max(float(np.max(np.hypot(u[i, nodes], v[i, nodes]))) for i in indices)
    fig, axes = plt.subplots(2, 1, figsize=(12., 5.), layout='constrained')
    speed = axes[0].tripcolor(display, np.hypot(u[indices[0], nodes], v[indices[0], nodes]),
                              shading='gouraud', cmap='viridis', vmin=0., vmax=speed_max)
    vorticity = axes[1].tripcolor(triangle, facecolors=_vorticity(basis, u[indices[0]], v[indices[0]]),
                                 cmap='RdBu_r', vmin=-20., vmax=20.)
    fig.colorbar(speed, ax=axes[0], label='P2 nodal speed', fraction=.025)
    fig.colorbar(vorticity, ax=axes[1], label='cell-average vorticity (±20)', fraction=.025)
    for ax in axes:
        ax.set(xlim=(0., 2.2), ylim=(0., .41), xlabel='x', ylabel='y')
        ax.set_aspect('equal')

    def frame(index: int) -> tuple:
        saved = indices[index]
        speed.set_array(np.hypot(u[saved, nodes], v[saved, nodes]))
        vorticity.set_array(_vorticity(basis, u[saved], v[saved]))
        axes[0].set_title(f'Saved accepted-node fields: t={actual[index]:.5f}')
        return speed, vorticity

    movie = FuncAnimation(fig, frame, frames=len(actual), interval=1000/15, blit=False)
    try:
        movie.save(directory/'wake.gif', writer=PillowWriter(fps=15), dpi=100)
    finally:
        plt.close(fig)
    return {'file': 'wake.gif', 'fps': 15, 'actual_times': actual.tolist(),
            'limits': 'nearest accepted nodes, no temporal interpolation; speed P2-node display, vorticity cell averages'}


def plot_reference_steps(fields: dict, directory: Path) -> None:
    has_scalar = 'scalar_history' in fields
    fig, axes = plt.subplots(2 if has_scalar else 1, 1, figsize=(10., 5. if has_scalar else 3.),
                              sharex=True, layout='constrained')
    axes = np.atleast_1d(axes)
    times = fields['times']
    rejected = ~fields['accepted']
    axes[0].semilogy(times[1:], np.diff(times), label=f'accepted: {len(times)-1}')
    if np.any(rejected):
        axes[0].semilogy(fields['attempt_t'][rejected], fields['attempt_h'][rejected], 'rx',
                         label=f'rejected: {int(np.sum(rejected))}')
    axes[0].set(ylabel='actual step', title=f'Saved step history; rejected trials: {int(np.sum(rejected))}')
    axes[0].legend()
    if has_scalar:
        axes[1].plot(times, fields['scalar_history'])
        axes[1].set_ylabel('SAV scalar r (saved history)')
    for ax in axes:
        ax.set_xlabel('time')
        ax.grid(alpha=.25)
    fig.savefig(directory/'steps.png', dpi=160)
    plt.close(fig)


def plot_performance(summary: dict, directory: Path) -> None:
    labels = ('baseline', 'direct', 'recycled')
    colors = ('#777777', '#0072B2', '#D55E00')
    fig, axes = plt.subplots(1, 2, figsize=(10., 4.), layout='constrained')
    for ax, mode in zip(axes, ('adaptive', 'replay')):
        values = [summary['table'][mode][label]['total_body_cpu_seconds'] for label in labels]
        medians = np.array([value['median'] for value in values])
        error = np.array([[value['median']-value['min'] for value in values],
                          [value['max']-value['median'] for value in values]])
        ax.bar(labels, medians, color=colors, yerr=error, capsize=4)
        ax.set(ylabel='CPU seconds (setup + integration + observations)',
               title=mode+'; median and observed min/max')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f"Identical native developed state; {summary['repeats']} interleaved repeats")
    fig.savefig(directory/'performance.png', dpi=160)
    plt.close(fig)


def plot_refinement(records: list[dict], directory: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10., 7.), layout='constrained')
    for ax, key in zip(axes.flat, ('drag_max', 'lift_max', 'strouhal', 'pressure_max')):
        for index, item in enumerate(records):
            ax.plot(index, item['cycles'][key], 'o', label=item['label'])
        ax.axhline(REFERENCE[key], color='k', linestyle='--', label='published numerical sample')
        ax.set(xticks=range(len(records)), xticklabels=[item['label'] for item in records], ylabel=key)
        ax.grid(alpha=.25)
        ax.tick_params(axis='x', rotation=15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Phase-free developed-cycle quantities; sampled reference is not a certified bracket')
    fig.savefig(directory/'refinement.png', dpi=160)
    plt.close(fig)

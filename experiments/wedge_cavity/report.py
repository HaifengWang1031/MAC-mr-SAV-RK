"""Read-only figures and animation for a recorded wedge-cavity run.

Nothing here mutates a record: every entry point loads `results.h5` through the
manifest checksum first, and analyses a copy of the stored nodal coefficients.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.records import PROJECT, digest
from solver.fem.wedge import WedgeSpace
from solver.fem.cavity import ClosedCavityModel
from solver.core import State
from . import fields
from .config import validate_stored

INK = '#23231F'
ACCENT = '#B4441F'
MUTED = '#8C8C88'


def load(directory: Path) -> tuple[dict, dict, dict]:
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('experiment') != 'wedge_cavity':
        raise ValueError('A complete wedge-cavity record is required')
    if digest(directory/'config.json') != manifest.get('config_sha256'):
        raise ValueError('Configuration checksum does not match the manifest')
    if digest(directory/'results.h5') != manifest.get('results_sha256'):
        raise ValueError('Results checksum does not match the manifest')
    # The record is re-derived from its own inputs, so a store that no longer
    # validates against them is rejected here rather than plotted.
    cfg = validate_stored(json.loads((directory/'config.json').read_text()))
    return cfg, manifest, {'directory': directory}


def open_run(directory: Path) -> tuple[dict, dict, WedgeSpace, ClosedCavityModel, 'RunData']:
    cfg, manifest, _ = load(directory)
    space = WedgeSpace(lid_length=cfg['lid_length'], apex_angle=cfg['apex_angle'],
                       lid_cells=cfg['lid_cells'], quadrature_order=cfg['quadrature_order'])
    model = ClosedCavityModel(space, cfg['nu'], lid_speed=cfg['lid_speed'],
                              # Analysis reconstructs the same FE representation,
                              # independently of optional time-solver libraries.
                              stokes_backend='direct', tolerance=cfg['stokes_tolerance'],
                              corner_policy=cfg['lid_corner_policy'])
    with h5py.File(Path(directory)/'results.h5', 'r') as data:
        if not np.array_equal(data['mesh/points'][:], space.mesh.p) \
                or not np.array_equal(data['mesh/triangles'][:], space.mesh.t.T):
            raise ValueError('Saved wedge mesh differs from the declared geometry')
        run = RunData(times=data['times'][:], kinetic=data['diagnostics/kinetic'][:],
                      snapshot_times=data['snapshots/times'][:],
                      u=data['snapshots/u'][:], v=data['snapshots/v'][:],
                      r=data['snapshots/r'][:],
                      pressure=data['snapshots/pressure'][:] if 'snapshots/pressure' in data else None,
                      pressure_indices=data['snapshots/pressure_indices'][:]
                      if 'snapshots/pressure_indices' in data else None)
    return cfg, manifest, space, model, run


@dataclass
class RunData:
    times: np.ndarray
    kinetic: np.ndarray
    snapshot_times: np.ndarray
    u: np.ndarray
    v: np.ndarray
    r: np.ndarray
    pressure: np.ndarray | None
    pressure_indices: np.ndarray | None

    def state(self, index: int) -> State:
        return State(float(self.snapshot_times[index]), self.u[index].copy(),
                     self.v[index].copy(), float(self.r[index]))

    def pressure_at(self, index: int) -> np.ndarray | None:
        if self.pressure is None or self.pressure_indices is None:
            return None
        position = np.flatnonzero(self.pressure_indices == index)
        return None if not position.size else self.pressure[int(position[0])]


def _mesh_outline(space: WedgeSpace) -> tuple[np.ndarray, np.ndarray]:
    mesh = space.mesh
    facet = mesh.facets[:, mesh.boundary_facets()]
    order = _boundary_order(mesh, facet)
    return mesh.p[0, order], mesh.p[1, order]


def _boundary_order(mesh, facets: np.ndarray) -> np.ndarray:
    """Chain boundary facets into a single closed polyline."""
    adjacency: dict[int, list[int]] = {}
    for a, b in facets.T:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    start = min(adjacency)
    order = [start]
    previous, current = start, adjacency[start][0]
    while current != start:
        order.append(current)
        following = [node for node in adjacency[current] if node != previous]
        if len(following) != 1:
            raise RuntimeError('Boundary is not a simple closed chain')
        previous, current = current, following[0]
        if len(order) > len(adjacency):
            raise RuntimeError('Boundary chain did not close')
    order.append(start)
    if len(order) != len(adjacency)+1:
        raise RuntimeError('Boundary chain missed nodes')
    return np.asarray(order)


def _style_axes(ax, space: WedgeSpace, *, label: bool = True) -> None:
    x, y = _mesh_outline(space)
    ax.plot(x, y, color=INK, lw=1.6, solid_capstyle='round', zorder=6)
    width = space.half_width
    ax.plot([-width, width], [0., 0.], color=ACCENT, lw=3.2, solid_capstyle='round', zorder=7)
    ax.annotate('', xy=(width*0.38, 0.012), xytext=(-width*0.38, 0.012),
                arrowprops=dict(arrowstyle='-|>', color=ACCENT, lw=1.6, mutation_scale=13), zorder=8)
    ax.set_aspect('equal')
    ax.set_xlim(-width*1.06, width*1.06)
    ax.set_ylim(-space.height*1.10, 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label:
        ax.text(width*0.44, -0.045*space.height, r'$U_0$', color=ACCENT, fontsize=8.5,
                ha='left', va='center', zorder=9)


def _panel(ax, sample: fields.Sample, values: np.ndarray, *, cmap: str, title: str,
           limits: tuple[float, float] | None = None, symmetric: bool = False,
           contours: bool = False, space: WedgeSpace | None = None,
           extend: str = 'both') -> tuple[Any, Any]:
    if limits is None:
        low, high = np.nanpercentile(values, .5), np.nanpercentile(values, 99.5)
        if symmetric:
            reach = max(abs(low), abs(high))
            limits = (-reach, reach)
        else:
            limits = (low, high)
    image = ax.tripcolor(sample.x, sample.y, sample.triangles, values, shading='gouraud',
                         cmap=cmap, vmin=limits[0], vmax=limits[1])
    if contours:
        psi = sample.streamfunction
        levels = fields.streamline_levels(psi)
        ax.tricontour(sample.x, sample.y, sample.triangles, psi, levels=levels,
                      colors='#2B2B28', linewidths=.55, alpha=.55)
    if space is not None:
        _style_axes(ax, space)
    ax.set_title(title, fontsize=10, color=INK, pad=4)
    bar = plt.colorbar(image, ax=ax, fraction=.042, pad=.02, extend=extend)
    bar.ax.tick_params(labelsize=7, length=2)
    bar.outline.set_linewidth(.5)
    bar.outline.set_edgecolor(MUTED)
    return image, bar


def final_figure(directory: Path, *, output: Path, sampler: fields.FieldSampler | None = None) -> dict:
    cfg, manifest, space, model, run = open_run(directory)
    sampler = sampler or fields.FieldSampler(model)
    index = len(run.snapshot_times)-1
    sample = sampler.sample(run.state(index), run.pressure_at(index))
    extrema = fields.vertex_extrema(sample, 5)
    if extrema:
        extrema[0] = fields.refine_extremum(sample, extrema[0])
    pressure = run.pressure_at(index)
    lid = model.boundary_force(model.vector(run.state(index)), pressure, 'lid') if pressure is not None else None

    figure, axes = plt.subplots(2, 2, figsize=(9.6, 8.0), layout='constrained')
    _panel(axes[0, 0], sample, sample.speed, cmap='viridis', title=r'$|\mathbf{u}|/U_0$',
           contours=True, space=space, extend='max')
    omega_reach = float(np.nanpercentile(np.abs(sample.vorticity), 88.))
    _panel(axes[0, 1], sample, sample.vorticity, cmap='RdBu_r',
           title=r'vorticity $\omega H/U_0$', limits=(-omega_reach, omega_reach), space=space)
    psi = sample.streamfunction
    psi_limits = (-max(abs(np.nanmin(psi)), abs(np.nanmax(psi))),
                  max(abs(np.nanmin(psi)), abs(np.nanmax(psi)))) \
        if np.nanmin(psi) < 0. < np.nanmax(psi) else (float(np.nanmin(psi)), float(np.nanmax(psi)))
    _panel(axes[1, 0], sample, psi, cmap='RdBu_r', title=r'streamfunction $\psi/(U_0 H)$',
           limits=psi_limits, contours=True, space=space)
    if sample.pressure is not None:
        _panel(axes[1, 1], sample, sample.pressure, cmap='coolwarm',
               title=r'pressure $p/(\rho U_0^2)$', symmetric=True, space=space)
    else:
        axes[1, 1].axis('off')

    for extrema_index, extremum in enumerate(extrema[:3]):
        marker = 'o' if extrema_index == 0 else 's'
        axes[1, 0].plot(extremum['x'], extremum['y'], marker, ms=5.0 if extrema_index == 0 else 3.4,
                        mfc='white' if extrema_index == 0 else 'none',
                        mec='#111111', mew=1.0, zorder=10)
    main = extrema[0] if extrema else None
    subtitle = (r'apex $2\alpha$ = ' + f'{cfg["apex_angle"]:.0f}°,  Re = {cfg["reynolds"]:.0f},  '
                f'lid cells = {cfg["lid_cells"]},  t = {run.snapshot_times[index]:.1f}')
    figure.suptitle('Lid-driven wedge cavity (SDIRK2-mr-ccSAV, P2/P1)\n'+subtitle,
                    fontsize=11, color=INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)
    return {'output': str(output), 'snapshot_index': int(index),
            'vortex_centres': extrema[:3],
            'streamfunction_min': float(np.nanmin(sample.streamfunction)),
            'vorticity_range': [float(np.nanmin(sample.vorticity)), float(np.nanmax(sample.vorticity))],
            'lid_force': None if lid is None else [float(lid[0]), float(lid[1])]}


def render_frames(directory: Path, *, output: Path, fps: int = 24, dpi: int = 165,
                  step: int = 1, hold: int = 12, sampler: fields.FieldSampler | None = None,
                  progress_every: int = 50, keep_frames: bool = False,
                  detail: str = 'vorticity') -> dict:
    from .animate import render
    return render(directory, output=output, fps=fps, dpi=dpi, step=step, hold=hold,
                  sampler=sampler, progress_every=progress_every, keep_frames=keep_frames, detail=detail)


def energy_figure(directory: Path, *, output: Path) -> dict:
    """Start-up history: kinetic energy and the strongest residual per accepted step."""
    cfg, manifest, _, _, run = open_run(directory)
    with h5py.File(Path(directory)/'results.h5', 'r') as data:
        residual = data['stage_residual'][:, 0] if 'stage_residual' in data else None
        divergence = data['stage_residual'][:, 1] if 'stage_residual' in data else None
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), layout='constrained')
    axes[0].plot(run.times, run.kinetic, color='#185FA5', lw=1.6)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel(r'$\frac{1}{2}\,\|\mathbf{u}\|^2$')
    axes[0].set_title('kinetic energy', fontsize=10)
    axes[0].grid(alpha=.22)
    if residual is not None:
        axes[1].semilogy(run.times[1:], np.maximum(residual, 1e-18), color=ACCENT, lw=.8,
                         label='stage residual')
        if divergence is not None:
            axes[1].semilogy(run.times[1:], np.maximum(divergence, 1e-18), color='#444441', lw=.8,
                             label=r'$\max|\nabla\!\cdot\!\mathbf{u}|$')
        axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel('t')
    axes[1].set_title('per-step checks', fontsize=10)
    axes[1].grid(alpha=.22, which='both')
    figure.suptitle(r'wedge cavity  $2\alpha$=' + f'{cfg["apex_angle"]:.0f}°  '
                    f'Re={cfg["reynolds"]:.0f}  lid cells={cfg["lid_cells"]}',
                    fontsize=10.5, color=INK)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {'output': str(output), 'final_kinetic': float(run.kinetic[-1])}


def main() -> None:
    parser = argparse.ArgumentParser(description='Figures and animation for a wedge-cavity run')
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--dpi', type=int, default=165)
    parser.add_argument('--skip-animation', action='store_true')
    parser.add_argument('--keep-frames', action='store_true')
    args = parser.parse_args()
    directory = args.directory
    out = args.out or Path(PROJECT)/'reports'/'wedge_cavity'/directory.name
    out.mkdir(parents=True, exist_ok=True)
    record: dict = {'run': directory.name, 'outputs': {}}
    record['outputs']['fields'] = final_figure(directory, output=out/'fields.png')
    record['outputs']['energy'] = energy_figure(directory, output=out/'energy.png')
    if not args.skip_animation:
        record['outputs']['animation'] = render_frames(directory, output=out/'wedge.mp4',
                                                       fps=args.fps, step=args.step, dpi=args.dpi,
                                                       keep_frames=args.keep_frames)
    (out/'report.json').write_text(json.dumps(record, indent=2, ensure_ascii=False, default=float)+'\n')
    print(json.dumps(record, indent=2, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()

"""Read-only comparison across recorded wedge-cavity runs.

Two studies share one table: apex-angle and Reynolds-number sweeps at fixed
resolution, and a mesh-refinement study at fixed physics. Every number is
recomputed from `results.h5`, so a stale report cannot outlive its record.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.workflow import PROJECT
from . import fields, report

INK = '#23231F'
ACCENT = '#B4441F'
MUTED = '#8C8C88'


def measure(directory: Path) -> dict:
    cfg, manifest, space, model, run = report.open_run(directory)
    sampler = fields.FieldSampler(model)
    index = len(run.snapshot_times)-1
    state = run.state(index)
    pressure = run.pressure_at(index)
    if pressure is None:
        raise ValueError('The selected snapshot has no recorded stage pressure for the force diagnostic')
    sample = sampler.sample(state, pressure)
    extrema = fields.vertex_extrema(sample, 4)
    refined = fields.refine_extremum(sample, extrema[0]) if extrema else None
    lid = model.boundary_force(model.vector(state), pressure, 'lid')
    metrics = manifest.get('metrics', {})
    return {
        'run': Path(directory).name, 'directory': str(directory), 'cfg': cfg,
        'apex_angle': cfg['apex_angle'], 'lid_cells': cfg['lid_cells'],
        'reynolds': cfg['reynolds'], 'element_size': cfg['element_size'],
        'final_time': float(run.snapshot_times[index]),
        'psi_min': float(np.nanmin(sample.streamfunction)),
        'centre': refined, 'secondary_extrema': extrema[1:3],
        'lid_drag': float(lid[0]), 'lid_lift': float(lid[1]),
        'kinetic': float(run.kinetic[-1]),
        'steady_rate': float(metrics.get('steady_rate_inf', float('nan'))),
        'divergence_inf': float(metrics.get('final_divergence_inf', float('nan'))),
        'max_stage_residual': float(metrics.get('max_stage_residual', float('nan'))),
        'streamfunction': sample.streamfunction, 'sample': sample, 'space': space,
        'model': model, 'velocity': model.vector(state),
    }


def _outline(ax, space, *, thin: bool = False) -> None:
    x, y = report._mesh_outline(space)
    ax.plot(x, y, color=INK, lw=1.2 if thin else 1.5)
    ax.plot([-space.half_width, space.half_width], [0., 0.], color=ACCENT, lw=2.4,
            solid_capstyle='round')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def montage(records: list[dict], *, output: Path, columns: int = 3,
            title: str = '', label=None) -> dict:
    rows = int(np.ceil(len(records)/columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.5*columns, 2.9*rows),
                                layout='constrained', squeeze=False)
    limits = (-max(abs(record['psi_min']) for record in records),
              max(abs(record['psi_min']) for record in records))
    for position, record in enumerate(records):
        ax = axes[position//columns][position % columns]
        sample = record['sample']
        image = ax.tripcolor(sample.x, sample.y, sample.triangles, sample.streamfunction,
                             shading='gouraud', cmap='RdBu_r', vmin=limits[0], vmax=limits[1])
        levels = np.linspace(record['psi_min'], 0., 26)[1:-1]
        ax.tricontour(sample.x, sample.y, sample.triangles, sample.streamfunction,
                      levels=levels, colors='#2B2B28', linewidths=.5, alpha=.55)
        refined = record['centre']
        if refined:
            ax.plot(refined['x'], refined['y'], 'o', ms=3.2, mfc='none', mec='#111111', mew=.8)
        _outline(ax, record['space'], thin=True)
        ax.set_title(label(record) if label else '', fontsize=8.5, color=INK, pad=7)
        bar = plt.colorbar(image, ax=ax, fraction=.045, pad=.02)
        bar.ax.tick_params(labelsize=6, length=1.5)
        bar.outline.set_linewidth(.4)
    for position in range(len(records), rows*columns):
        axes[position//columns][position % columns].axis('off')
    figure.suptitle(title, fontsize=11, color=INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return {'output': str(output)}


def convergence_figure(records: list[dict], *, output: Path, coordinate: float = 0.0) -> dict:
    """Vertical centreline u(0, y) and the mesh-to-mesh increment, for a fixed problem."""
    ordered = sorted(records, key=lambda record: record['lid_cells'])
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout='constrained')
    profiles = []
    for record in ordered:
        space = record['space']
        y = np.linspace(-space.height, 0., 401)
        points = np.vstack([np.full_like(y, coordinate), y])
        velocity = record['model'].evaluate_velocity(record['velocity'], points)
        profiles.append((record['lid_cells'], y, velocity[0]))
        axes[0].plot(velocity[0], y/space.height, '-', lw=1.5,
                     label=f"{record['lid_cells']} lid cells")
    axes[0].axvline(0., color=MUTED, ls=':', lw=.8)
    axes[0].set_xlabel(r'$u(0,y)/U_0$')
    axes[0].set_ylabel(r'$y/H$')
    axes[0].set_title('vertical centreline velocity', fontsize=10)
    axes[0].grid(alpha=.22)
    axes[0].legend(fontsize=7.5, frameon=False)

    fine = profiles[-1]
    depth = -fine[1]/ordered[-1]['space'].height
    for cells, y, u in profiles[:-1]:
        coarser = np.interp(fine[1], y, u)
        axes[1].semilogy(depth, np.abs(fine[2]-coarser) + 1e-16,
                         lw=1.3, label=f'{cells} vs {fine[0]}')
    axes[1].set_xlabel(r'$-y/H$  (depth below the lid)')
    axes[1].set_ylabel(r'$|u_{\mathrm{fine}}-u_{\mathrm{coarse}}|/U_0$')
    axes[1].set_title('mesh-to-mesh increment', fontsize=10)
    axes[1].grid(alpha=.22, which='both')
    axes[1].legend(fontsize=7.5, frameon=False)
    figure.suptitle(r'wedge cavity, $2\alpha$ = 90$^\circ$, Re = 100: mesh refinement', fontsize=11, color=INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {'output': str(output), 'cells': [record['lid_cells'] for record in ordered]}


def table(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        centre = record['centre'] or {}
        rows.append({'run': record['run'], 'apex_angle': record['apex_angle'],
                     'lid_cells': record['lid_cells'], 'reynolds': record['reynolds'],
                     'psi_min': record['psi_min'],
                     'x_centre': centre.get('x'), 'y_centre': centre.get('y'),
                     'lid_drag': record['lid_drag'], 'lid_lift': record['lid_lift'],
                     'kinetic': record['kinetic'], 'steady_rate': record['steady_rate'],
                     'divergence_inf': record['divergence_inf']})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare wedge-cavity runs')
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, default=Path(PROJECT)/'reports'/'wedge_cavity')
    parser.add_argument('--groups', type=str, default='angle,reynolds,grid')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    records = {record['run']: record for record in (measure(directory) for directory in args.directories)}
    # The animation run is the same problem sampled far more densely; it is not an
    # independent data point for any sweep, so it never enters a comparison group.
    ordered = sorted((record for record in records.values()
                      if record['cfg'].get('case') != 'animation'),
                     key=lambda record: (record['apex_angle'], record['reynolds'],
                                         record['lid_cells']))

    def _best_per_key(items: list[dict], key) -> list[dict]:
        """Keep the finest mesh for each key, so a sweep never shows one case twice."""
        chosen: dict = {}
        for record in items:
            current = chosen.get(key(record))
            if current is None or record['lid_cells'] > current['lid_cells']:
                chosen[key(record)] = record
        return sorted(chosen.values(), key=lambda record: record['lid_cells'])

    summary: dict = {'table': table(ordered), 'figures': {}}
    groups = set(args.groups.split(','))
    if 'angle' in groups:
        angle = _best_per_key([record for record in ordered
                               if record['reynolds'] == 100. and record['apex_angle'] in (60., 90., 120.)],
                              lambda record: record['apex_angle'])
        angle.sort(key=lambda record: record['apex_angle'])
        if len(angle) >= 2:
            summary['figures']['angle'] = montage(
                angle, output=out/'study_apex_angle.png',
                title=r'apex angle sweep at Re = 100 (streamlines, vortex centre marked)',
                label=lambda record: r'$2\alpha$' + f" = {record['apex_angle']:.0f}°, "
                                     + r'$\psi_{\min}$' + f" = {record['psi_min']:.4f}")
    if 'reynolds' in groups:
        reynolds = _best_per_key([record for record in ordered
                                  if record['apex_angle'] == 90. and record['reynolds'] in (10., 100., 400.)],
                                 lambda record: record['reynolds'])
        reynolds.sort(key=lambda record: record['reynolds'])
        if len(reynolds) >= 2:
            summary['figures']['reynolds'] = montage(
                reynolds, output=out/'study_reynolds.png',
                title=r'Reynolds sweep at $2\alpha$ = 90° (streamlines, vortex centre marked)',
                label=lambda record: f"Re = {record['reynolds']:.0f}, "
                                     + r'$\psi_{\min}$' + f" = {record['psi_min']:.4f}")
    if 'grid' in groups:
        grid = _best_per_key([record for record in ordered
                              if record['apex_angle'] == 90. and record['reynolds'] == 100.],
                             lambda record: record['lid_cells'])
        if len(grid) >= 2:
            summary['figures']['grid'] = convergence_figure(grid, output=out/'study_grid.png')
            finest = grid[-1]
            summary['grid_deltas'] = [
                {'cells': record['lid_cells'],
                 'd_psi_min': abs(record['psi_min']-finest['psi_min']),
                 'd_lid_drag': abs(record['lid_drag']-finest['lid_drag']),
                 'd_x_centre': abs((record['centre'] or {}).get('x', np.nan)
                                   - (finest['centre'] or {}).get('x', np.nan)),
                 'd_y_centre': abs((record['centre'] or {}).get('y', np.nan)
                                   - (finest['centre'] or {}).get('y', np.nan))}
                for record in grid]
    (out/'study.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                             default=lambda value: None)+'\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=lambda value: None))


if __name__ == '__main__':
    main()

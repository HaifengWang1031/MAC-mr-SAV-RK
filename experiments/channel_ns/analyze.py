"""Read explicit channel records; analysis never constructs a model or advances time."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from solver.channel import ChannelGrid
from experiments.workflow import PROJECT, digest, provenance, write_json
from .problems import DiscreteManufactured, Manufactured, poiseuille, startup
from .run import load_channel


def read_fields(directory: Path) -> tuple[dict, dict[str, Any]]:
    cfg, manifest = load_channel(directory)
    grid = ChannelGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    with h5py.File(directory/'results.h5', 'r') as data:
        u, v = data['final/u'][:], data['final/v'][:]
        centre_u, centre_v = .5*(u[:, :-1]+u[:, 1:]), .5*(v[:-1]+v[1:])
        # omega lives on interior y vertices and unique periodic x vertices.
        omega = (v[1:-1]-np.roll(v[1:-1], 1, axis=1))/grid.hx \
            -np.diff(u[:, :-1], axis=0)/grid.hy
        return cfg, {'u': u, 'v': v, 'speed': np.hypot(centre_u, centre_v),
                     'vorticity': omega, 'profile': np.mean(u[:, :-1], axis=1),
                     'flow_rates': grid.hy*np.sum(u[:, :-1], axis=0),
                     'times': data['times'][:], 'dt': data['dt'][:],
                     'final_time': float(data['final/t'][()]), 'r': float(data['final/r'][()]),
                     'status': manifest['status'],
                     'diagnostics': {key: data['diagnostics/'+key][:] for key in data['diagnostics']},
                     'requested_times': data['snapshots/requested_times'][:],
                     'actual_times': data['snapshots/actual_times'][:],
                     'max_stage_residual': float(np.max(data['stages/residual'][:], initial=0.)),
                     'max_scalar_residual': float(np.max(data['stages/scalar_residual'][:], initial=0.))}


def velocity_error(cfg: dict, fields: dict) -> tuple[float, float, str] | None:
    grid = ChannelGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    t = fields['final_time']
    if cfg['case'] in ('mms', 'discrete_mms'):
        cls = DiscreteManufactured if cfg['case'] == 'discrete_mms' else Manufactured
        exact = cls(grid, cfg['nu'], cfg['amplitude']).exact(t)
        label = 'fixed_grid_discrete_mms' if cfg['case'] == 'discrete_mms' else 'continuous_channel_mms'
    elif cfg['case'] == 'poiseuille':
        exact = poiseuille(grid, cfg['nu'], cfg['driving'])
        label = 'continuous_poiseuille'
    elif cfg['case'] == 'startup':
        exact = startup(grid, cfg['nu'], cfg['driving'], t)
        label = 'continuous_startup_diffusion_series'
    else:
        return None
    w = grid.pack(fields['u'], fields['v'])
    return grid.norm(w-exact), grid.norm(exact), label


def plot_fields(cfg: dict, fields: dict, report: Path) -> None:
    """Optional XeLaTeX export; face averages use the actual MAC locations."""
    import matplotlib as mpl
    from matplotlib.figure import Figure
    with mpl.rc_context({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Times New Roman'],
                         'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
                         'pgf.preamble': r'\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{amsmath}',
                         'text.latex.preamble': r'\usepackage{amsmath}'}):
        figure = Figure(figsize=(9, 6), layout='constrained')
        axes = figure.subplots(2, 2)
        x, y = np.linspace(0., cfg['lx'], cfg['nx']+1), np.linspace(0., cfg['ly'], cfg['ny']+1)
        im = axes[0, 0].pcolormesh(x, y, fields['speed'], cmap='magma', shading='flat', rasterized=True)
        figure.colorbar(im, ax=axes[0, 0])
        axes[0, 0].set(xlabel='$x$', ylabel='$y$', title=r'$|\boldsymbol{u}|$', aspect='equal')
        yc = (np.arange(cfg['ny'])+.5)*cfg['ly']/cfg['ny']
        axes[0, 1].plot(fields['profile'], yc, label='Computed')
        if cfg['case'] in ('poiseuille', 'startup'):
            grid = ChannelGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
            exact = poiseuille(grid, cfg['nu'], cfg['driving']) if cfg['case'] == 'poiseuille' else \
                startup(grid, cfg['nu'], cfg['driving'], fields['final_time'])
            axes[0, 1].plot(grid.unpack(exact)[0][:, 0], yc, '--', label='Continuous reference')
            axes[0, 1].legend()
        axes[0, 1].set(xlabel=r'$\overline{u}(y)$', ylabel='$y$', ylim=(0., cfg['ly']))
        axes[1, 0].plot(fields['times'], fields['diagnostics']['kinetic'])
        axes[1, 0].set(xlabel='$t$', ylabel=r'$\frac12\|\boldsymbol{u}\|_h^2$')
        axes[1, 1].plot(fields['times'], fields['diagnostics']['flow_rate'])
        axes[1, 1].set(xlabel='$t$', ylabel='$Q$')
        figure.suptitle(f"{cfg['case']}, {cfg['scheme']}, {fields['status']}, $t={fields['final_time']:.6g}$")
        figure.savefig(report/'fields.pdf', backend='pgf')
        figure.savefig(report/'fields.png', backend='pgf', dpi=150)


def analyze(directory: Path, *, root: Path = PROJECT, plot: bool = False) -> Path:
    directory = directory.resolve()
    cfg, fields = read_fields(directory)
    _, manifest = load_channel(directory)
    report = root/'reports'/'channel_ns'/directory.name
    report.mkdir(parents=True, exist_ok=True)
    summary = {'run': str(directory), 'run_id': directory.name, 'status': fields['status'],
               'error': manifest.get('error', ''), 'final_time': fields['final_time'],
               'accepted_steps': len(fields['dt']), 'metrics': manifest['metrics'],
               'manifest_sha256': digest(directory/'manifest.json'), 'analysis_source': provenance(),
               'field_layout': cfg['field_layout'], 'pressure_semantics': cfg['pressure_semantics'],
               'max_stage_residual': fields['max_stage_residual'],
               'max_scalar_residual': fields['max_scalar_residual'],
               'max_physical_divergence': float(np.max(fields['diagnostics']['divergence_inf'])),
               'flow_rate_spread': float(np.ptp(fields['flow_rates'])), 'r_final': fields['r'],
               'snapshot_requested_times': fields['requested_times'].tolist(),
               'snapshot_actual_times': fields['actual_times'].tolist()}
    error = velocity_error(cfg, fields)
    if error is not None:
        absolute, norm, label = error
        summary.update(velocity_L2_error=absolute, velocity_relative_L2_error=absolute/norm if norm else None,
                       reference=label+'_at_actual_final_time')
    else:
        summary['reference'] = 'perturbed_channel_diagnostics_only'
    if cfg['case'] == 'poiseuille':
        grid = ChannelGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
        summary['discrete_equilibrium_L2_error'] = grid.norm(
            grid.pack(fields['u'], fields['v'])-poiseuille(grid, cfg['nu'], cfg['driving'], discrete=True))
    if cfg['case'] in ('mms', 'discrete_mms'):
        summary.update(scalar_absolute_error=abs(fields['r']), scalar_reference=0.)
    write_json(report/'analysis.json', summary)
    names = list(fields['diagnostics'])
    with (report/'history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['time', 'dt', *names])
        for index, time in enumerate(fields['times']):
            writer.writerow([time, 0. if index == 0 else fields['dt'][index-1],
                             *[fields['diagnostics'][name][index] for name in names]])
    with (report/'mean_profile.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['y', 'mean_u'])
        writer.writerows(zip((np.arange(cfg['ny'])+.5)*cfg['ly']/cfg['ny'], fields['profile']))
    if plot: plot_fields(cfg, fields, report)
    (report/'analysis.log').write_text(f"{fields['status']}: read saved run {directory.name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    if args.validation:
        from .validate import analyze_validation
        print(analyze_validation(args.run, root=args.root))
    else:
        print(analyze(args.run, root=args.root, plot=args.plot))


if __name__ == '__main__': main()

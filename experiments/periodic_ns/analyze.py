"""Analyze explicit saved periodic records. No model construction or integration."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from solver.fourier import FourierSpace
from experiments.workflow import PROJECT, digest, write_json
from .problems import Manufactured
from .run import load_periodic


def read_fields(directory: Path) -> tuple[dict, dict[str, Any]]:
    """Collocated fields: no MAC face averaging, no finite-difference vorticity."""
    cfg, manifest = load_periodic(directory)
    s = FourierSpace(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    with h5py.File(directory/'results.h5', 'r') as stored:
        velocity = np.stack([stored['final/u'][:], stored['final/v'][:]])
        z = s.to_spectral(velocity)
        fields: dict[str, Any] = {
            'u': velocity[0], 'v': velocity[1], 'speed': np.hypot(*velocity),
            'vorticity': s.to_physical(1j*(s.kx*z[1]-s.ky*z[0])),
            'final_time': float(stored['final/t'][()]), 'r': float(stored['final/r'][()]),
            'times': stored['times'][:], 'dt': stored['dt'][:], 'status': manifest['status'],
            'diagnostics': {key: stored['diagnostics/'+key][:] for key in stored['diagnostics']},
            'requested_times': stored['snapshots/requested_times'][:],
            'actual_times': stored['snapshots/actual_times'][:],
        }
        fields['max_stage_residual'] = float(np.max(stored['stages/residual'][:], initial=0.))
        fields['max_scalar_residual'] = float(np.max(stored['stages/scalar_residual'][:], initial=0.))
    return cfg, fields


def plot_fields(cfg: dict, fields: dict, report: Path) -> None:
    """Optional export with XeLaTeX and embedded Times New Roman text."""
    import matplotlib as mpl
    from matplotlib.figure import Figure
    with mpl.rc_context({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Times New Roman'],
               'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
               'pgf.preamble': r'\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{amsmath}',
               'text.latex.preamble': r'\usepackage{amsmath}'}):
        figure = Figure(figsize=(9, 7), layout='constrained')
        axes = figure.subplots(2, 2)
        # Include the wrapped endpoint only in the display mesh: samples remain
        # at x=j*hx,y=i*hy, rather than being shifted by half a cell as in imshow.
        x = np.linspace(0., cfg['lx'], cfg['nx']+1)
        y = np.linspace(0., cfg['ly'], cfg['ny']+1)
        for axis, name, label in ((axes[0, 0], 'speed', r'$|\boldsymbol{u}|$'),
                                  (axes[0, 1], 'vorticity', r'$\omega=\partial_xv-\partial_yu$')):
            limit = float(np.max(np.abs(fields[name])))
            im = axis.pcolormesh(x, y, np.pad(fields[name], ((0, 1), (0, 1)), mode='wrap'),
                                 cmap='magma' if name == 'speed' else 'RdBu_r', shading='nearest',
                                 vmin=0. if name == 'speed' else -limit, vmax=limit, rasterized=True)
            figure.colorbar(im, ax=axis)
            axis.set(title=label, xlabel='$x$', ylabel='$y$',
                     xlim=(0., cfg['lx']), ylim=(0., cfg['ly']), aspect='equal')
        axes[1, 0].plot(fields['times'], fields['diagnostics']['kinetic'])
        axes[1, 0].set(xlabel='$t$', ylabel=r'$\frac12\|\boldsymbol{u}\|_{L^2}^2$')
        axes[1, 1].plot(fields['times'], fields['diagnostics']['r'])
        axes[1, 1].set(xlabel='$t$', ylabel='$r$')
        figure.suptitle(f"{cfg['scheme']}, {fields['status']}, $t={fields['final_time']:.6g}$")
        figure.savefig(report/'fields.pdf', backend='pgf')
        figure.savefig(report/'fields.png', backend='pgf', dpi=150)


def analyze(directory: Path, *, root: Path = PROJECT, plot: bool = False) -> Path:
    directory = directory.resolve()
    cfg, fields = read_fields(directory)
    _, manifest = load_periodic(directory)
    report = root/'reports'/'periodic_ns'/directory.name
    report.mkdir(parents=True, exist_ok=True)
    summary = {'run': str(directory), 'run_id': directory.name, 'status': fields['status'],
               'error': manifest.get('error', ''), 'final_time': fields['final_time'],
               'accepted_steps': len(fields['dt']), 'metrics': manifest['metrics'],
               'manifest_sha256': digest(directory/'manifest.json'),
               'field_layout': cfg['field_layout'], 'max_stage_residual': fields['max_stage_residual'],
               'max_scalar_residual': fields['max_scalar_residual'],
               'max_physical_divergence': float(np.max(fields['diagnostics']['divergence_inf'])),
               'r_final': fields['r'], 'pressure_semantics': cfg['pressure_semantics'],
               'snapshot_requested_times': fields['requested_times'].tolist(),
               'snapshot_actual_times': fields['actual_times'].tolist()}
    if cfg['case'] == 'mms':
        s = FourierSpace(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
        exact = Manufactured(s, cfg['nu'], cfg['amplitude']).exact(fields['final_time'])
        difference = np.stack([fields['u'], fields['v']])-exact
        error = float(np.sqrt(s.area*np.mean(np.sum(difference**2, axis=0))))
        norm = float(np.sqrt(s.area*np.mean(np.sum(exact**2, axis=0))))
        summary.update(velocity_L2_error=error, velocity_relative_L2_error=error/norm if norm > 0 else None,
                       scalar_absolute_error=abs(fields['r']), scalar_reference=0.,
                       reference='analytic_continuous_periodic_mms_at_actual_final_time')
    else:
        summary['reference'] = 'no_exact_reference; diagnostics_only'
    write_json(report/'analysis.json', summary)
    names = list(fields['diagnostics'])
    with (report/'history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time', 'dt', *names])
        for index, time in enumerate(fields['times']):
            writer.writerow([time, 0. if index == 0 else fields['dt'][index-1],
                             *[fields['diagnostics'][name][index] for name in names]])
    if plot:
        plot_fields(cfg, fields, report)
    (report/'analysis.log').write_text(f"{fields['status']}: read saved run {directory.name}\n")
    return report


def analyze_validation(directory: Path, *, root: Path = PROJECT) -> Path:
    """Verify study artifacts and all explicit child runs, then aggregate repeats."""
    directory = directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('experiment') != 'periodic_ns_validation':
        raise ValueError('Complete periodic validation study required')
    if digest(directory/'config.json') != manifest['config_sha256']:
        raise ValueError('Validation config checksum mismatch')
    for name, expected in manifest['artifacts'].items():
        if digest(directory/name) != expected:
            raise ValueError(f'Validation artifact checksum mismatch: {name}')
    cfg = json.loads((directory/'config.json').read_text())
    results = json.loads((directory/'results.json').read_text())
    if len(results['benchmark']) != len(cfg['benchmark_sizes'])*len(cfg['schemes'])*2*cfg['repeats'] \
            or len(results['temporal']) != len(cfg['schemes'])*len(cfg['gammas']) \
            or len(results['spatial']) != len(cfg['spatial_sizes']):
        raise ValueError('Incomplete periodic validation matrix')
    for row in results['temporal']:
        errors = []
        for member in row['members']:
            child = directory/member
            child_cfg, saved = load_periodic(child, require_complete=True)
            if saved['source'] != manifest['source']:
                raise ValueError('Validation mixes source/environment identities')
            _, fields = read_fields(child)
            s = FourierSpace(child_cfg['nx'], child_cfg['ny'], child_cfg['lx'], child_cfg['ly'])
            delta = np.stack([fields['u'], fields['v']])-Manufactured(s, child_cfg['nu']).exact(fields['final_time'])
            errors.append(float(np.sqrt(s.area*np.mean(np.sum(delta**2, axis=0)))))
        if not np.allclose(errors, row['velocity_L2_errors'], rtol=1e-12, atol=1e-15):
            raise ValueError('Temporal saved-field error mismatch')
    for row in results['spatial']:
        with np.load(directory/f"stokes_{row['size']}.npz", allow_pickle=False) as fields:
            delta = fields['velocity']-fields['exact']
            error = float(np.sqrt((2*np.pi)**2*np.mean(np.sum(delta**2, axis=0))))
            if not np.isclose(error, row['velocity_L2_error'], rtol=1e-12, atol=1e-15):
                raise ValueError('Spatial saved-field error mismatch')
    groups: dict[tuple[int, str, str], list[dict]] = {}
    for row in results['benchmark']:
        child_cfg, saved = load_periodic(directory/row['member'], require_complete=True)
        if saved['source'] != manifest['source']:
            raise ValueError('Validation mixes source/environment identities')
        if child_cfg['nx'] != row['size'] or child_cfg['scheme'] != row['scheme'] \
                or any(row[key] != value for key, value in saved['metrics'].items()):
            raise ValueError('Benchmark member metadata mismatch')
        groups.setdefault((row['size'], row['scheme'], row['schedule']), []).append(row)
    timings = []
    for (n, name, schedule), rows in groups.items():
        if {row['repeat'] for row in rows} != set(range(cfg['repeats'])) or len(rows) != cfg['repeats']:
            raise ValueError('Missing/duplicate benchmark repeat')
        totals = [r['setup_cpu_seconds']+r['integration_cpu_seconds'] for r in rows]
        timings.append({'size': n, 'scheme': name, 'schedule': schedule, 'repeats': len(rows),
                        'setup_cpu_seconds': float(np.median([r['setup_cpu_seconds'] for r in rows])),
                        'integration_cpu_seconds': float(np.median([r['integration_cpu_seconds'] for r in rows])),
                        'total_cpu_seconds': float(np.median(totals)),
                        'total_cpu_min': min(totals), 'total_cpu_max': max(totals),
                        'max_stage_residual': max(r['max_stage_residual'] for r in rows),
                        'max_physical_divergence': max(r['max_physical_divergence'] for r in rows)})
    profiles = []
    for n in cfg['benchmark_sizes']:
        for component in ('nonlinear', 'stokes_single_rhs', 'stokes_two_rhs', 'diagnostics'):
            rows = [r for r in results['profiles'] if r['size'] == n and r['component'] == component]
            if len(rows) != cfg['repeats'] or {r['repeat'] for r in rows} != set(range(cfg['repeats'])):
                raise ValueError('Missing/duplicate component timing')
            profiles.append({'size': n, 'component': component,
                             'cpu_seconds_per_call': float(np.median([r['cpu_seconds_per_call'] for r in rows]))})
    report = root/'reports'/'periodic_ns_validation'/directory.name
    report.mkdir(parents=True, exist_ok=True)
    from solver.schemes.registry import scheme_info
    observed = all(min(row['velocity_rates']) > scheme_info(row['scheme']).order-.25
                   for row in results['temporal'])
    write_json(report/'analysis.json', {'run': str(directory), 'manifest_sha256': digest(directory/'manifest.json'),
                                       'temporal': results['temporal'], 'spatial': results['spatial'],
                                       'timings': timings, 'profiles': profiles, 'limits': results['limits'],
                                       'expected_velocity_orders_observed': observed,
                                       'thread_environment': manifest['source']['threads']})
    for name, rows in (('timings.csv', timings), ('components.csv', profiles)):
        with (report/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    print(analyze_validation(args.run, root=args.root) if args.validation else
          analyze(args.run, root=args.root, plot=args.plot))


if __name__ == '__main__':
    main()

"""Compare saved Orszag–Tang runs in time and against digitized paper curves."""
import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from experiments.records import PROJECT, digest, write_json
from .records import load_mhd

PAPER = Path(__file__).parent/'reference'/'orszag_tang_1979.csv'
QUANTITIES = {'kinetic': 'Kinetic energy', 'magnetic': 'Magnetic energy',
              'total_energy': 'Total energy', 'dissipation': 'Total dissipation'}
LABELS = {'sdirk3': 'SDIRK3', 'sdirk3_mrsav': 'SDIRK3-mr-ccSAV'}


@dataclass(frozen=True)
class SavedRun:
    path: Path
    config: dict[str, Any]
    source: dict[str, Any]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fields(record: SavedRun) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    with h5py.File(record.path/'results.h5', 'r') as data:
        return {float(t): (np.stack((u, v)), b) for t, u, v, b in
                zip(data['snapshots/actual_times'], data['snapshots/u'],
                    data['snapshots/v'], data['snapshots/magnetic'])}


def _errors(fields: tuple[np.ndarray, np.ndarray],
            exact: tuple[np.ndarray, np.ndarray]) -> list[float]:
    return [float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a, b in zip(fields, exact)]


def _time_comparison(reference: SavedRun, check: SavedRun, runs: list[SavedRun],
                     output: Path) -> dict:
    configs = [item.config for item in (check, *runs)]
    keys = ('nx', 'ny', 'nu', 'eta', 'amplitude', 'T', 'snapshots')
    if any(any(cfg[key] != reference.config[key] for key in keys) for cfg in configs) \
            or any(r.config['dt'] <= reference.config['dt'] for r in (check, *runs)):
        raise ValueError('Time comparison requires a finer reference on the same grid and times')
    ref = _fields(reference)
    check_fields = _fields(check)
    rows: list[dict] = []
    summary: dict[str, Any] = {}
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    terminal = reference.config['T']
    for t in ref:
        if t == 0.:
            continue
        sensitivity = _errors(check_fields[t], ref[t])
        time_summary: dict[str, Any] = {'reference_relative_change_u_B': sensitivity, 'schemes': {}}
        summary[f'{t:g}'] = time_summary
        for name, label in LABELS.items():
            group = sorted((r for r in runs if r.config['scheme'] == name),
                           key=lambda r: r.config['dt'], reverse=True)
            if len(group) < 2 or len({r.config['dt'] for r in group}) != len(group):
                raise ValueError(f'Need distinct time steps for {name}')
            steps = np.asarray([r.config['dt'] for r in group])
            errors = np.asarray([_errors(_fields(r)[t], ref[t]) for r in group])
            rates = np.log(errors[:-1]/errors[1:])/np.log(steps[:-1, None]/steps[1:, None])
            for j, (run, dt, (err_u, err_b)) in enumerate(zip(group, steps, errors)):
                rows.append({'time': t, 'scheme': name, 'dt': dt, 'relative_l2_u': err_u,
                             'relative_l2_B': err_b, 'order_u': rates[j-1, 0] if j else '',
                             'order_B': rates[j-1, 1] if j else '', 'run_id': run.path.name})
            time_summary['schemes'][name] = {
                'finest_pair_order_u_B': rates[-1].tolist(),
                'finest_relative_error_u_B': errors[-1].tolist(),
                'reference_change_over_finest_error_u_B': (sensitivity/errors[-1]).tolist(),
            }
            if t == terminal:
                for ax, values in zip(axes, errors.T):
                    ax.loglog(steps, values, 'o-', label=label)
        if t == terminal:
            for ax, error, field in zip(axes, errors[-1], ('Velocity', 'Magnetic field')):
                ax.loglog(steps, .1*error*(steps/steps[-1])**3, '--', color='0.4', label='Slope 3')
                ax.set_xticks(steps, labels=[f'{dt:.4g}' for dt in steps])
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set(title=f'{field} at t = {terminal:g}', xlabel='Time step',
                       ylabel='Relative L2 error')
                ax.legend()
    _write_csv(output/'time_errors.csv', rows)
    figure.savefig(output/'time_convergence.png', dpi=180)
    plt.close(figure)
    return summary


def _curves(record: SavedRun) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(record.path/'results.h5', 'r') as data:
        initial_energy = data['diagnostics/total_energy'][0]
        return data['times'][:], {key: data[f'diagnostics/{key}'][:]/initial_energy for key in QUANTITIES}


def _curve_comparison(runs: list[SavedRun], grid_check: SavedRun, step_check: SavedRun,
                      output: Path, suffix: str) -> dict:
    if sorted(r.config['scheme'] for r in runs) != sorted(LABELS):
        raise ValueError('Paper comparison needs one run of each scheme')
    baseline = next(r for r in runs if r.config['scheme'] == 'sdirk3')
    common = ('nu', 'eta', 'amplitude', 'T')
    if (baseline.config['nu'], baseline.config['eta'], baseline.config['T']) != (.02, .02, 2.) \
            or any(any(r.config[k] != baseline.config[k] for k in common)
                   for r in (*runs, grid_check, step_check)) \
            or any(r.config['scheme'] != 'sdirk3' for r in (grid_check, step_check)) \
            or grid_check.config['dt'] != baseline.config['dt'] \
            or (step_check.config['nx'], step_check.config['ny']) != (baseline.config['nx'], baseline.config['ny']) \
            or step_check.config['dt'] <= baseline.config['dt'] \
            or any(r.config[k] != baseline.config[k] for r in runs for k in ('nx', 'ny', 'dt')):
        raise ValueError('Paper comparison requires matching Fig. 1(c)/5 parameters and paired checks')
    with PAPER.open(newline='') as stream:
        paper = list(csv.DictReader(stream))
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout='constrained')
    amplitude = baseline.config['amplitude']
    figure.suptitle(f'Orszag–Tang · initial amplitude factor {amplitude:g}')
    rows: list[dict] = []
    summary: dict[str, Any] = {'paper_points_sha256': digest(PAPER)}
    cached = {record.path: _curves(record) for record in (*runs, grid_check, step_check)}
    baseline_times, baseline_values = cached[baseline.path]
    for ax, (key, label) in zip(axes.flat, QUANTITIES.items()):
        points = [point for point in paper if point['quantity'] == key]
        paper_t = np.array([float(point['time']) for point in points])
        paper_y = np.array([float(point['normalized_value']) for point in points])
        ax.plot(paper_t, paper_y, 'o', color='0.25', ms=3.5, label=f'Orszag–Tang Fig. {points[0]["figure"]}')
        stats: dict[str, Any] = {}
        summary[key] = stats
        for record in runs:
            name = record.config['scheme']
            t, curves = cached[record.path]
            values = curves[key]
            compared = np.interp(paper_t, t, values)
            difference = compared-paper_y
            stats[name] = {'max_abs_vs_paper': float(np.max(np.abs(difference))),
                           'rms_vs_paper': float(np.sqrt(np.mean(difference**2)))}
            if key == 'dissipation':
                peak, paper_peak = np.argmax(values), np.argmax(paper_y)
                stats[name].update(peak_time=float(t[peak]), peak=float(values[peak]),
                                   paper_sample_peak_time=float(paper_t[paper_peak]),
                                   paper_sample_peak=float(paper_y[paper_peak]))
            ax.plot(t, values, '--' if name.endswith('mrsav') else '-', lw=1.2, label=LABELS[name])
            for pt, observed, computed in zip(paper_t, paper_y, compared):
                rows.append({'quantity': key, 'time': pt, 'scheme': name, 'run_id': record.path.name,
                             'paper_normalized': observed, 'computed_normalized': computed})
        for role, record in [('grid_check', grid_check), ('step_check', step_check)]:
            t, curves = cached[record.path]
            stats[role+'_max_abs_change'] = float(
                np.max(np.abs(np.interp(t, baseline_times, baseline_values[key])-curves[key])))
        ax.set(xlim=(0, 2), title=label, xlabel='Time', ylabel='Normalized by initial total energy')
        ax.legend(fontsize=8)
    _write_csv(output/f'paper_comparison{suffix}.csv', rows)
    figure.savefig(output/f'paper_comparison{suffix}.png', dpi=180)
    plt.close(figure)
    return summary


def analyze(batch_file: Path, output: Path) -> dict:
    batch = json.loads(batch_file.read_text())
    if batch['status'] != 'complete':
        raise ValueError('Complete study required')
    groups: dict[str, list[SavedRun]] = {}
    for member in batch['members']:
        path = Path(member['run'])
        config, manifest = load_mhd(path, require_complete=True)
        groups.setdefault(member['role'], []).append(SavedRun(path, config, manifest['source']))
    for role in ('reference', 'reference_check'):
        if len(groups.get(role, [])) != 1:
            raise ValueError(f'Expected one {role} run')
    reference = groups['reference'][0]
    if any(r.source != reference.source for group in groups.values() for r in group):
        raise ValueError('All members must use the same source and environment')
    output.mkdir(parents=True, exist_ok=True)
    time = _time_comparison(reference, groups['reference_check'][0], groups['convergence'], output)
    curves = {}
    for amplitude in sorted({r.config['amplitude'] for r in groups['curves']}):
        selected = [r for r in groups['curves'] if r.config['amplitude'] == amplitude]
        grid_check = [r for r in groups['curve_grid_check'] if r.config['amplitude'] == amplitude]
        step_check = [r for r in groups['curve_step_check'] if r.config['amplitude'] == amplitude]
        if len(grid_check) != 1 or len(step_check) != 1:
            raise ValueError(f'Expected one grid and step check at amplitude {amplitude:g}')
        suffix = '' if amplitude == reference.config['amplitude'] else f'_amplitude_{amplitude:g}'
        curves[f'{amplitude:g}'] = _curve_comparison(selected, grid_check[0], step_check[0],
                                                    output, suffix)
    summary = {'batch': str(batch_file.resolve()), 'time_convergence': time, 'paper_curves': curves}
    write_json(output/'summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or PROJECT/'reports'/'periodic_mhd'/args.batch.parent.name
    analyze(args.batch, output)
    print(output)


if __name__ == '__main__':
    main()

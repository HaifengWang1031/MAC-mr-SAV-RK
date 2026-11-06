"""Compare two saved Zhou S0=125 runs with Fig. 2 and inspect merger diagnostics."""
import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from experiments.records import PROJECT, digest, write_json
from solver.fourier import FourierSpace
from .analyze import LABELS, _write_csv
from .records import load_mhd
from .zhou import ZHOU_CASE

PAPER = Path(__file__).parent/'reference'/'zhou_2019_fig2_s125.csv'


def snapshot_metrics(space: FourierSpace, flux: np.ndarray,
                     magnetic: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Count prominent sampled O points and compute the magnetic shell spectrum."""
    limit = .1*np.max(np.abs(flux))
    maxima = (flux == maximum_filter(flux, size=3, mode='wrap')) & (flux > limit)
    minima = (flux == minimum_filter(flux, size=3, mode='wrap')) & (flux < -limit)
    mode_energy = .5*space.weights*np.sum(np.abs(space.to_spectral(magnetic))**2, axis=0)
    shells = np.rint(np.sqrt(space.k2)).astype(int)
    spectrum = np.bincount(shells.ravel(), weights=mode_energy.ravel())
    return int(maxima.sum()+minima.sum()), int(1+np.argmax(spectrum[1:])), spectrum


def _merger_windows(rows: list[dict]) -> dict:
    """Bracket the first count crossing sustained for at least three frames."""
    counts = [row['o_points'] for row in rows]
    return {str(target): next(
        ([rows[i-1]['time'], rows[i]['time']] for i in range(1, len(rows)-2)
         if counts[i-1] > target and max(counts[i:i+3]) <= target), None)
        for target in (128, 64, 32, 16, 8, 4)}


def analyze_zhou(ordinary: Path, sav: Path, output: Path) -> dict:
    records = [load_mhd(p, require_complete=True) for p in (ordinary, sav)]
    configs = [cfg for cfg, _ in records]
    base = configs[0]
    if [cfg['scheme'] for cfg in configs] != list(LABELS) or any(
            {k: v for k, v in cfg.items() if k != 'scheme'} !=
            {k: v for k, v in base.items() if k != 'scheme'} for cfg in configs[1:]):
        raise ValueError('Use paired ordinary and SAV runs with identical controls')
    if base['case'] != ZHOU_CASE or base['S0'] != 125 or base['snapshots'][0] != 0:
        raise ValueError('Fig. 2 comparison requires the Zhou S0=125 case with an initial snapshot')
    if records[0][1]['source'] != records[1][1]['source']:
        raise ValueError('The paired runs must have the same source and environment')
    frame_times = [t for t in (0, 12, 20, 40, 80) if t in base['snapshots']]
    with h5py.File(ordinary/'results.h5', 'r') as left, h5py.File(sav/'results.h5', 'r') as right:
        if any(not np.array_equal(left[f'snapshots/{key}'][0], right[f'snapshots/{key}'][0])
               for key in ('u', 'v', 'magnetic')):
            raise ValueError('The saved initial fields differ')
        limits = {time: max(np.max(np.abs(record['snapshots/current'][base['snapshots'].index(time)]))
                            for record in (left, right)) for time in frame_times}

    output.mkdir(parents=True, exist_ok=True)
    space = FourierSpace(base['nx'], base['ny'])
    tau0 = np.sqrt(base['S0'])*np.pi**2/32
    with PAPER.open(newline='') as stream:
        paper = list(csv.DictReader(stream))
    paper_t, paper_energy = np.array([[float(p[key]) for key in paper[0]] for p in paper]).T
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    energy_ax, shifted_ax, island_ax, peak_ax = axes.flat
    shifted_ax.loglog(paper_t, paper_energy, color='.3', lw=2, label='Zhou Fig. 2, shifted time')
    maps, map_axes = plt.subplots(2, len(frame_times), squeeze=False,
                                  figsize=(2.7*len(frame_times), 5.3), layout='constrained')
    maps.suptitle('Magnetic flux contours over current density')
    summary: dict = {'S0': base['S0'], 'grid': [base['nx'], base['ny']], 'dt': base['dt'],
                     'T': base['T'], 'tau0': float(tau0), 'paper_points_sha256': digest(PAPER),
                     'paper_clock': '(t-t0)/tau0', 'simulation_clock': 't/tau0, no fitted shift',
                     'shared_initial_fields_verified': True, 'schemes': {}}
    all_rows: list[dict] = []
    paper_rows: list[dict] = []
    energies: list[np.ndarray] = []
    for row, (directory, cfg) in enumerate(zip((ordinary, sav), configs)):
        scheme = cfg['scheme']
        label = LABELS[scheme]
        style = '-' if scheme == 'sdirk3' else '--'
        points: list[dict] = []
        with h5py.File(directory/'results.h5', 'r') as data:
            t, energy = data['times'][:], data['diagnostics/magnetic'][:]
            energies.append(energy)
            normalized = energy/energy[0]
            energy_ax.semilogy(t, normalized, style, label=label)
            shifted_ax.loglog(t[1:]/tau0, energy[1:], style, label=label)
            overlap = paper_t <= t[-1]/tau0
            compare = np.interp(paper_t[overlap], t/tau0, energy)
            for x, observed, computed in zip(paper_t[overlap], paper_energy[overlap], compare):
                paper_rows.append({'scheme': scheme, 'time_over_tau0': x, 'paper_energy': observed,
                                   'computed_energy_unshifted': computed})
            for i, time in enumerate(data['snapshots/actual_times']):
                flux = data['snapshots/flux'][i]
                count, peak, spectrum = snapshot_metrics(space, flux, data['snapshots/magnetic'][i])
                current = data['snapshots/current'][i]
                points.append({'scheme': scheme, 'time': float(time), 'o_points': count,
                               'peak_k': peak, 'peak_shell_energy': float(spectrum[peak]),
                               'flux_max': float(np.max(flux)), 'current_max_abs': float(np.max(np.abs(current)))})
                if time in frame_times:
                    column = frame_times.index(time)
                    ax = map_axes[row, column]
                    limit = limits[time]
                    background = ax.imshow(current, extent=(0, space.lx, 0, space.ly), origin='lower',
                                           cmap='RdBu_r', vmin=-limit, vmax=limit)
                    ax.contour(space.x, space.y, flux, levels=12, colors='black', linewidths=.32)
                    ax.set(title=f't = {time:g}' if row == 0 else '', xticks=[], yticks=[])
                    if column == 0:
                        ax.set_ylabel(label)
                    if row == 1:
                        maps.colorbar(background, ax=map_axes[:, column], orientation='horizontal',
                                      label='Current density', shrink=.8)
        all_rows.extend(points)
        times = [point['time'] for point in points]
        island_ax.step(times, [point['o_points'] for point in points],
                       where='post', linestyle=style, label=label)
        peak_ax.step(times, [point['peak_k'] for point in points],
                     where='post', linestyle=style, label=label)
        summary['schemes'][scheme] = {'run': str(directory.resolve()), 'final_EB_over_EB0': float(normalized[-1]),
                                      'final_o_points': points[-1]['o_points'],
                                      'peak_current_over_saved_frames': max(p['current_max_abs'] for p in points),
                                      'peak_k_at_saved_frames': {str(p['time']): p['peak_k'] for p in points
                                                                 if p['time'] in frame_times},
                                      'o_point_crossing_windows': _merger_windows(points)}
    summary['max_energy_difference_over_EB0'] = float(np.max(np.abs(energies[0]-energies[1]))/energies[0][0])
    energy_ax.set(xlabel='Simulation time', ylabel='Magnetic energy / initial')
    shifted_ax.set(xlabel='Time / initial reconnection time', ylabel='Magnetic energy density',
                   xlim=(.1, max(25., base['T']/tau0)))
    island_ax.set(xlabel='Simulation time', ylabel='Prominent sampled O points', yscale='log')
    peak_ax.set(xlabel='Simulation time', ylabel='Magnetic spectrum peak k')
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(output/'zhou_diagnostics.png', dpi=170)
    maps.savefig(output/'zhou_snapshots.png', dpi=170)
    plt.close(fig)
    plt.close(maps)
    _write_csv(output/'snapshot_diagnostics.csv', all_rows)
    if paper_rows:
        _write_csv(output/'figure2_comparison.csv', paper_rows)
    write_json(output/'summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ordinary', type=Path)
    parser.add_argument('sav', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ordinary, sav = (p if p.is_absolute() else PROJECT/'runs'/'periodic_mhd'/p
                     for p in (args.ordinary, args.sav))
    analyze_zhou(ordinary, sav, args.output)
    print(args.output)


if __name__ == '__main__':
    main()

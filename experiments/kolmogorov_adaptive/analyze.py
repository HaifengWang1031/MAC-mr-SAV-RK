"""Read-only six-panel I/PI comparison for the no-slip Kolmogorov case."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.workflow import PROJECT, digest, write_json
from solver.mac.grid import MACGrid

SCHEMES = ('sdirk2', 'sdirk2_mrsav', 'sdirk3', 'sdirk3_mrsav')
COLORS = {'sdirk2': 'C0', 'sdirk2_mrsav': 'C4',
          'sdirk3': '#2E8B57', 'sdirk3_mrsav': 'C6'}
STYLES = {'I': ('-', 'o'), 'PI': ('--', '^'), 'fixed': (':', 's')}


def read_member(path: Path) -> tuple[dict, dict, dict]:
    config = json.loads((path/'config.json').read_text())
    manifest = json.loads((path/'manifest.json').read_text())
    if hashlib.sha256((path/'results.npz').read_bytes()).hexdigest() != manifest['result_sha256']:
        raise ValueError(f'Checksum mismatch: {path}')
    with np.load(path/'results.npz', allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    return config, manifest, values


def aligned_output_times(config: dict, data: dict, label: str) -> np.ndarray:
    count = round(config['T']/config['output_every'])
    expected = config['output_every']*np.arange(count+1)
    actual = data['output_times']
    if actual.shape != expected.shape or not np.allclose(
            actual, expected, atol=1e-8, rtol=0):
        raise ValueError(f'Output times are incompatible: {label}')
    return expected


def analyze_batch(batch_path: Path, *, root: Path = PROJECT) -> Path:
    batch = json.loads(batch_path.read_text())
    adaptive = batch['members']
    if len(adaptive) != 8 or batch.get('fixed_members') or not batch['reference']:
        raise ValueError('Eight adaptive runs and one reference run are required')
    if {(item['scheme'], item['controller']) for item in adaptive} != {
            (scheme, controller) for scheme in SCHEMES for controller in ('I', 'PI')}:
        raise ValueError('Incomplete adaptive comparison')
    config = batch['config']
    if 'atol_velocity' not in config or 'rtol_velocity' not in config:
        raise ValueError('This analysis requires velocity-controlled runs')
    grid = MACGrid(config['nx'], config['ny'], 2*np.pi, 2*np.pi)
    records = []
    source_hash = batch['source']['code_sha256']
    for item in adaptive:
        actual, manifest, data = read_member(Path(item['path']))
        if manifest['status'] != 'complete' or manifest['source']['code_sha256'] != source_hash \
                or any(actual[key] != config[key] for key in config):
            raise ValueError('Mixed, failed, or stale adaptive member')
        records.append((item, manifest, data))
    reference_config, reference_manifest, reference = read_member(Path(batch['reference']))
    if reference_manifest['status'] != 'complete' or \
            reference_manifest['source']['code_sha256'] != source_hash or any(
                reference_config[key] != config[key] for key in config):
        raise ValueError('Reference is failed, stale, or incompatible')
    folder = root/'reports/kolmogorov_adaptive'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    (folder/'figures').mkdir(parents=True)
    (folder/'tables').mkdir()
    analysis: dict = {'status': 'running', 'batch': str(batch_path.resolve()),
                      'batch_sha256': digest(batch_path),
                      'inputs': [item['path'] for item, _, _ in records],
                      'reference': batch['reference']}
    write_json(folder/'analysis.json', analysis)
    try:
        nominal_outputs = aligned_output_times(config, reference, 'reference')
        ref_velocity = [grid.pack(u, v) for u, v in zip(
            reference['output_u'], reference['output_v'], strict=True)]
        plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                             'font.size': 12})
        fig, panels = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=False)
        axes = panels.ravel()
        summary = []
        for item, manifest, data in records:
            scheme, controller = item['scheme'], item['controller']
            label = f'{scheme} {controller}'
            color = COLORS[scheme]
            linestyle, marker = STYLES[controller]
            t, h = data['attempt_t'], data['attempt_h']
            accepted = data['attempt_accepted']
            cpu = data['attempt_cpu']
            markevery = max(1, len(t)//10)
            line = {'color': color, 'linestyle': linestyle, 'linewidth': 1.05,
                    'marker': marker, 'markevery': markevery, 'markersize': 3.5,
                    'markeredgewidth': 0, 'label': label}
            accepted_times = (t+h)[accepted]
            axes[0].step(np.r_[0., accepted_times],
                         np.r_[h[accepted][0] if np.any(accepted) else np.nan, h[accepted]],
                         where='post', **line)
            axes[1].plot(np.r_[0., accepted_times],
                         np.arange(len(accepted_times)+1), **line)
            axes[2].plot(np.r_[0., accepted_times], np.r_[0., cpu[accepted]], **line)
            if 'mrsav' in scheme:
                axes[3].plot(data['times'], np.abs(data['diagnostic_r']), **line)
            if controller != 'fixed':
                valid = np.isfinite(data['attempt_error']) & (data['attempt_error'] > 0)
                accepted_valid = valid & accepted
                axes[4].semilogy((t+h)[accepted_valid],data['attempt_error'][accepted_valid],**line)
                rejected_valid = valid & ~accepted
                axes[4].scatter((t+h)[rejected_valid],data['attempt_error'][rejected_valid],
                                marker='x',s=20,color=color,zorder=5)
            times = aligned_output_times(config, data, item['path'])
            errors = []
            for index in range(len(times)):
                velocity = grid.pack(data['output_u'][index], data['output_v'][index])
                denominator = max(grid.norm(ref_velocity[index]), 1e-14)
                errors.append(grid.norm(velocity-ref_velocity[index])/denominator)
            axes[5].semilogy(times, errors, **line)
            final_error = errors[-1] if len(times) and abs(times[-1]-config['T']) < 1e-10 else float('nan')
            cpu_seconds = float(cpu[-1]) if cpu.size else float('nan')
            summary.append({'scheme': scheme, 'controller': controller,
                            'status': manifest['status'], 'accepted': manifest['accepted_steps'],
                            'rejected': manifest['rejected_trials'],
                            'wall_seconds': manifest['seconds'], 'cpu_seconds': cpu_seconds,
                            'final_ref_error': final_error,
                            'max_abs_r': float(np.max(np.abs(data['diagnostic_r'])))})
        ylabels = ['Step sizes', 'Step count', 'CPU time (s)', r'$|r|$',
                   r'$L^2$ embedded error / tolerance',
                   r'Relative $L^2$ error of velocity']
        for index, axis in enumerate(axes):
            axis.set(xlabel=r'$t$', ylabel=ylabels[index])
            axis.grid(True, color='0.8', linestyle='--', linewidth=.6)
            axis.text(.5, -.30, f'({chr(ord("a") + index)})',
                      transform=axis.transAxes, ha='center', va='top', fontsize=15)
        axes[0].set_yscale('log')
        for bound in (config['min_step'], config['max_step']):
            axes[0].axhline(bound, color='0.35', linestyle='--', linewidth=.8)
        axes[4].axhline(1., color='black', linestyle='--', linewidth=.8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
                   bbox_to_anchor=(.5, .005))
        fig.subplots_adjust(left=.06, right=.95, bottom=.20, top=.98,
                            wspace=.38, hspace=.62)
        fig.savefig(folder/'figures/adaptive_comparison.pdf', bbox_inches='tight')
        fig.savefig(folder/'figures/adaptive_comparison.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
        with (folder/'tables/summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        analysis['status'] = 'complete'
    except Exception as exc:
        analysis.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(folder/'analysis.json', analysis)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(analyze_batch(args.batch, root=args.root))

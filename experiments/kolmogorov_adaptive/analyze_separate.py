"""One six-panel comparison per adaptive run with three same-scheme fixed controls."""
import argparse
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
from experiments.kolmogorov_adaptive.analyze import SCHEMES, read_member
from solver.mac.grid import MACGrid


def analyze_separate(batch_path: Path, *, root: Path = PROJECT) -> Path:
    batch = json.loads(batch_path.read_text())
    steps = batch['fixed_steps']
    if len(batch['members']) != 8 or len(batch['fixed_members']) != 12 or \
            sorted(steps) != sorted([.005, .001, .0005]):
        raise ValueError('Expected eight adaptive runs and three fixed steps per scheme')
    if batch['status'] != 'complete':
        raise ValueError('Fixed-control batch did not complete')
    config = batch['config']
    grid = MACGrid(config['nx'], config['ny'], 2*np.pi, 2*np.pi)
    _, reference_manifest, reference = read_member(Path(batch['reference']))
    if reference_manifest['status'] != 'complete':
        raise ValueError('Reference run did not complete')
    reference_velocity = {round(float(time), 12): grid.pack(u, v) for time, u, v in zip(
        reference['output_times'], reference['output_u'], reference['output_v'])}
    folder = root/'reports/kolmogorov_adaptive'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    (folder/'figures').mkdir(parents=True)
    analysis = {'status': 'running', 'batch': str(batch_path.resolve()),
                'batch_sha256': digest(batch_path), 'reference': batch['reference'],
                'figures': []}
    write_json(folder/'analysis.json', analysis)
    try:
        plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                             'font.size': 12})
        for adaptive in batch['members']:
            scheme, controller = adaptive['scheme'], adaptive['controller']
            if scheme not in SCHEMES or controller not in ('I', 'PI'):
                raise ValueError('Unexpected adaptive member')
            controls = [item for item in batch['fixed_members'] if item['scheme'] == scheme]
            if sorted(item['step'] for item in controls) != sorted(steps):
                raise ValueError(f'Incomplete fixed controls for {scheme}')
            entries = [(f'{scheme} {controller}', adaptive, 'C0', '-', 'o')]
            entries.extend((rf'Fixed $\tau={item["step"]:g}$', item, color, '--', marker)
                           for item, color, marker in zip(
                               sorted(controls, key=lambda value: -value['step']),
                               ('C1', 'C2', 'C3'), ('*', 'd', 's')))
            fig, panels = plt.subplots(2, 3, figsize=(18, 10))
            axes = panels.ravel()
            for label, item, color, linestyle, marker in entries:
                actual, manifest, data = read_member(Path(item['path']))
                if manifest['status'] != 'complete' or any(
                        actual[key] != value for key, value in config.items()
                        if key != 'fixed_step'):
                    raise ValueError(f'Incompatible or failed member: {item["path"]}')
                t, h = data['attempt_t'], data['attempt_h']
                accepted = data['attempt_accepted']
                times = (t+h)[accepted]
                style = {'label': label, 'color': color, 'linestyle': linestyle,
                         'marker': marker, 'markevery': max(1, len(t)//18),
                         'markersize': 3.5, 'linewidth': 1.1}
                axes[0].step(np.r_[0., times], np.r_[h[accepted][0], h[accepted]],
                             where='post', **style)
                axes[1].plot(np.r_[0., times], np.arange(len(times)+1), **style)
                axes[2].plot(np.r_[0., times],
                             np.r_[0., data['attempt_cpu'][accepted]], **style)
                if 'mrsav' in scheme:
                    axes[3].plot(data['times'], np.abs(data['diagnostic_r']), **style)
                if item is adaptive:
                    valid = np.isfinite(data['attempt_error']) & (data['attempt_error'] > 0)
                    axes[4].semilogy((t+h)[valid & accepted],
                                     data['attempt_error'][valid & accepted], **style)
                    axes[4].scatter((t+h)[valid & ~accepted],
                                    data['attempt_error'][valid & ~accepted],
                                    color=color, marker='x', s=20)
                errors = []
                for time, u, v in zip(data['output_times'], data['output_u'], data['output_v']):
                    velocity = grid.pack(u, v)
                    target = reference_velocity[round(float(time), 12)]
                    errors.append(grid.norm(velocity-target)/max(grid.norm(target), 1e-14))
                axes[5].semilogy(data['output_times'], errors, **style)
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
            axes[4].axhline(1., color='0.35', linestyle='--', linewidth=.8)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=11,
                       bbox_to_anchor=(.5, .01))
            fig.subplots_adjust(left=.08, right=.97, bottom=.16, top=.98,
                                wspace=.34, hspace=.52)
            stem = f'{scheme}-{controller}'
            fig.savefig(folder/f'figures/{stem}.pdf', bbox_inches='tight')
            fig.savefig(folder/f'figures/{stem}.png', dpi=180, bbox_inches='tight')
            plt.close(fig)
            analysis['figures'].append(stem)
            write_json(folder/'analysis.json', analysis)
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
    print(analyze_separate(args.batch, root=args.root))

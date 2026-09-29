"""Read-only six-panel figures for completed natural-grid adaptive members."""
import argparse
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
from experiments.kolmogorov_adaptive.analyze import read_member, aligned_output_times
from solver.mac.grid import MACGrid

PHYSICAL_KEYS = ('nx', 'ny', 'nu', 'm', 'epsilon', 'initial_modes', 'gamma', 'T', 'output_every')


def matching_physics(actual: dict, expected: dict) -> bool:
    return all(actual[key] == expected[key] for key in PHYSICAL_KEYS)


def fixed_controls(selection: dict, scheme: str) -> list[dict]:
    base = json.loads(Path(selection['base_batch']).read_text())
    controls = [item for item in base['fixed_members'] if item['scheme'] == scheme]
    controls += [item for item in selection['fixed_005'] if item['scheme'] == scheme]
    controls.sort(key=lambda item: -(item.get('step', .005)))
    steps = [item.get('step', .005) for item in controls]
    if steps != [.005, .0025, .001, .0005]:
        raise ValueError(f'Incomplete fixed controls: {scheme}, {steps}')
    return controls


def main(campaign_path: Path, selection_path: Path) -> Path:
    campaign = json.loads(campaign_path.read_text())
    selection = json.loads(selection_path.read_text())
    completed = [item for item in campaign['members'] if item['status'] == 'complete'
                 and item.get('reference_verified') is True]
    if not completed:
        raise ValueError('No complete member with a verified reference')
    if selection['status'] != 'complete':
        raise ValueError('Fixed-control selection is incomplete')
    config = campaign['base_config']
    grid = MACGrid(config['nx'], config['ny'], 2*np.pi, 2*np.pi)
    ref_config, ref_manifest, fixed_reference = read_member(Path(selection['reference']))
    if ref_manifest['status'] != 'complete' or not matching_physics(ref_config, config):
        raise ValueError('Fixed-control reference is incompatible')
    fixed_ref_times = aligned_output_times(config, fixed_reference, 'fixed reference')
    fixed_ref_velocity = [grid.pack(u, v) for u, v in zip(
        fixed_reference['output_u'], fixed_reference['output_v'], strict=True)]

    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory = PROJECT/'reports/kolmogorov_adaptive/natural_step_partial'/identity
    figures = directory/'figures'
    figures.mkdir(parents=True)
    record_path = directory/'analysis.json'
    record = {
        'status': 'running', 'campaign': str(campaign_path.resolve()),
        'campaign_sha256_at_analysis': digest(campaign_path),
        'selection': str(selection_path.resolve()),
        'fixed_reference': selection['reference'],
        'note': 'Adaptive errors use each natural accepted node and an independently refined reference; fixed errors use outputs every 0.1 and the previously saved SDIRK3 reference.',
        'figures': [],
    }
    write_json(record_path, record)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11})
    try:
        for member in completed:
            scheme, controller = member['scheme'], member['controller']
            actual, manifest, adaptive = read_member(Path(member['path']))
            if manifest['status'] != 'complete' or not matching_physics(actual, config):
                raise ValueError(f'Adaptive member is incompatible: {member["path"]}')
            error_path = Path(member['reference_error_path'])
            if hashlib.sha256(error_path.read_bytes()).hexdigest() != member['reference_error_sha256']:
                raise ValueError(f'Adaptive reference error checksum mismatch: {error_path}')
            with np.load(error_path, allow_pickle=False) as source:
                error = {key: source[key] for key in source.files}
            if not np.array_equal(error['times'], adaptive['times']):
                raise ValueError(f'Adaptive error times differ: {scheme}-{controller}')
            fig, panels = plt.subplots(2, 3, figsize=(18, 10))
            axes = panels.ravel()
            entries = [(f'{scheme} {controller}', adaptive, 'C0', '-', 'o', True,
                        actual, error)]
            for j, (control, marker) in enumerate(zip(
                    fixed_controls(selection, scheme), ('*', 'd', 's', '^')), start=1):
                fixed_config, fixed_manifest, data = read_member(Path(control['path']))
                if fixed_manifest['status'] != 'complete' or not matching_physics(
                        fixed_config, config):
                    raise ValueError(f'Fixed control is incompatible: {control["path"]}')
                entries.append((rf'Fixed $\tau={control.get("step", .005):g}$', data,
                                f'C{j}', '--', marker, False, fixed_config, None))
            for label, data, color, linestyle, marker, is_adaptive, settings, errors in entries:
                t, h = data['attempt_t'], data['attempt_h']
                accepted = data['attempt_accepted']
                node_times = (t+h)[accepted]
                style = {'label': label, 'color': color, 'linestyle': linestyle,
                         'marker': marker, 'markevery': max(1, len(node_times)//18),
                         'markersize': 3.3, 'linewidth': 1.1}
                axes[0].step(np.r_[0., node_times], np.r_[h[accepted][0], h[accepted]],
                             where='post', **style)
                axes[1].plot(np.r_[0., node_times], np.arange(len(node_times)+1), **style)
                cpu = data['attempt_cpu'][accepted]
                valid_cpu = np.isfinite(cpu)
                if np.any(valid_cpu):
                    axes[2].plot(np.r_[0., node_times[valid_cpu]],
                                 np.r_[0., cpu[valid_cpu]], **style)
                if 'mrsav' in scheme:
                    axes[3].plot(data['times'], np.abs(data['diagnostic_r']), **style)
                if is_adaptive:
                    accepted_error = data['attempt_error'][accepted]
                    kinetic = data['diagnostic_kinetic']
                    if len(kinetic) != len(accepted_error)+1:
                        raise ValueError('Accepted kinetic history is misaligned')
                    old_norm = np.sqrt(np.maximum(0., 2*kinetic[:-1]))
                    new_norm = np.sqrt(np.maximum(0., 2*kinetic[1:]))
                    raw_embedded = accepted_error*(settings['atol_velocity']+
                        settings['rtol_velocity']*np.maximum(old_norm, new_norm))
                    positive = np.isfinite(raw_embedded) & (raw_embedded > 0)
                    axes[4].semilogy(node_times[positive], raw_embedded[positive], **style)
                    positive_error = errors['relative_velocity_l2'] > 0
                    axes[5].semilogy(errors['times'][positive_error],
                                     errors['relative_velocity_l2'][positive_error], **style)
                    positive_defect = errors['relative_reference_difference'] > 0
                    axes[5].semilogy(errors['times'][positive_defect],
                                     errors['relative_reference_difference'][positive_defect],
                                     color='0.55', linestyle=':', linewidth=.8,
                                     label='Reference refinement difference')
                else:
                    output_times = aligned_output_times(config, data, label)
                    if not np.allclose(output_times, fixed_ref_times, rtol=0, atol=1e-8):
                        raise ValueError('Fixed and reference outputs do not match')
                    relative = [grid.norm(grid.pack(u, v)-target)/max(grid.norm(target), 1e-14)
                                for u, v, target in zip(data['output_u'], data['output_v'],
                                                        fixed_ref_velocity, strict=True)]
                    positive = np.asarray(relative) > 0
                    axes[5].semilogy(output_times[positive], np.asarray(relative)[positive],
                                     **style)
            labels = ('Step sizes', 'Accepted-step count', 'Adaptive/fixed CPU time (s)',
                      r'$|r|$', r'$L^2$ embedded difference',
                      r'Relative velocity $L^2$ error')
            for j, ylabel in enumerate(labels):
                axes[j].set(xlabel=r'$t$', ylabel=ylabel)
                axes[j].grid(True, color='0.82', linestyle='--', linewidth=.6)
                axes[j].text(.5, -.29, f'({chr(ord("a")+j)})', transform=axes[j].transAxes,
                             ha='center', va='top', fontsize=14)
            axes[0].set_yscale('log')
            axes[0].axhline(config['max_step'], color='0.4', linestyle=':', linewidth=.8)
            axes[4].set_yscale('log')
            if 'mrsav' not in scheme:
                axes[3].set_axis_off()
                axes[3].text(.5, .5, 'No auxiliary variable',
                             transform=axes[3].transAxes, ha='center', va='center',
                             color='0.4', fontsize=14)
            handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='lower center', ncol=5,
                       fontsize=10, bbox_to_anchor=(.5, .01))
            fig.suptitle(f'{scheme} {controller}: natural adaptive nodes', y=.995)
            fig.subplots_adjust(left=.08, right=.97, bottom=.16, top=.94,
                                wspace=.34, hspace=.55)
            stem = f'{scheme}-{controller}'
            png = figures/f'{stem}.png'
            pdf = figures/f'{stem}.pdf'
            fig.savefig(png, dpi=180, bbox_inches='tight')
            fig.savefig(pdf, bbox_inches='tight')
            plt.close(fig)
            record['figures'].append({'scheme': scheme, 'controller': controller,
                                      'png': str(png), 'pdf': str(pdf)})
            write_json(record_path, record)
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(record_path, record)
    print(directory)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    main(args.campaign, args.selection)

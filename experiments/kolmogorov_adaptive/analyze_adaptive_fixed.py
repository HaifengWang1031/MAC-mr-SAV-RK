"""Read-only six-panel comparison of joint adaptivity and same-scheme fixed runs."""
import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.kolmogorov_adaptive.analyze import read_member
from experiments.workflow import PROJECT, digest, provenance, write_json
from solver.mac.grid import MACGrid


def sample_indices(times: np.ndarray, queries: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(queries, times).clip(0, len(queries)-1)
    left = (indices-1).clip(0, len(queries)-1)
    indices = np.where(abs(queries[left]-times) <= abs(queries[indices]-times), left, indices)
    if np.any(abs(queries[indices]-times) > 1e-8):
        raise ValueError('Snapshot times do not match shared reference queries')
    return indices


def accepted_embedded_difference(data: dict[str, np.ndarray], *, atol: float,
                                 rtol: float) -> tuple[np.ndarray, np.ndarray]:
    """Undo velocity normalization using K=||u||_h^2/2 at accepted nodes.

    Rejected trial velocities are not persisted, so only accepted differences
    can be recovered from the saved kinetic energies.
    """
    accepted = data['attempt_accepted'].astype(bool)
    kinetic = data['diagnostic_kinetic']
    normalized = data['attempt_error_u'][accepted]
    if kinetic.shape != data['times'].shape or len(kinetic) != len(normalized)+1:
        raise ValueError('Accepted errors and kinetic-energy nodes are not aligned')
    if (not np.isfinite(kinetic).all() or np.any(kinetic < 0)
            or not np.isfinite(normalized).all() or np.any(normalized < 0)):
        raise ValueError('Invalid accepted velocity errors or kinetic energies')
    np.testing.assert_allclose((data['attempt_t']+data['attempt_h'])[accepted],
                               data['times'][1:], rtol=0, atol=1e-10)
    norms = np.sqrt(2*kinetic)
    scale = atol+rtol*np.maximum(norms[:-1], norms[1:])
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid velocity normalization scales')
    return normalized*scale, scale


def rejected_embedded_difference(directory: Path, member: Path,
                                 data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Load a completed replay tied to this exact representative run."""
    replay = json.loads((directory/'manifest.json').read_text())
    original = json.loads((member/'manifest.json').read_text())
    if (replay['status'] != 'complete' or Path(replay['member']).resolve() != member.resolve()
            or replay['result_sha256'] != original['result_sha256']
            or replay['source'] != original['source']
            or replay['config_sha256'] != digest(member/'config.json')
            or replay['diagnostic_sha256'] != digest(directory/'rejected_difference.npz')):
        raise ValueError('Incompatible or corrupt rejected-trial replay')
    with np.load(directory/'rejected_difference.npz') as archive:
        values = {key: archive[key] for key in archive.files}
    indices = np.flatnonzero(~data['attempt_accepted'].astype(bool))
    np.testing.assert_array_equal(values['attempt_index'], indices)
    np.testing.assert_allclose(values['trial_time'],
                               (data['attempt_t']+data['attempt_h'])[indices], rtol=0, atol=1e-10)
    for key, original_key in (('error_u', 'attempt_error_u'), ('error_r', 'attempt_error_r')):
        np.testing.assert_allclose(values[key], data[original_key][indices], rtol=1e-9, atol=1e-11)
    difference = values['velocity_l2_embedded_difference']
    if not np.isfinite(difference).all() or np.any(difference < 0):
        raise ValueError('Invalid rejected embedded differences')
    return values


def relative_embedded_difference(difference: np.ndarray, trial_norm: np.ndarray) -> np.ndarray:
    """Divide the embedded difference by the candidate endpoint velocity norm."""
    if (difference.shape != trial_norm.shape or not np.isfinite(trial_norm).all()
            or np.any(trial_norm < 0) or not np.isfinite(difference).all() or np.any(difference < 0)):
        raise ValueError('Invalid relative embedded-difference inputs')
    if np.any((trial_norm == 0) & (difference != 0)):
        raise ValueError('Nonzero embedded difference relative to zero trial velocity')
    return np.divide(difference, trial_norm, out=np.zeros_like(difference), where=trial_norm > 0)


def analyze(campaign_path: Path, *, root: Path = PROJECT,
            rejected_replay: Path | None = None, preliminary_coarse: bool = False) -> Path:
    campaign = json.loads(campaign_path.read_text())
    if (not campaign['members'] or not campaign['references']
            or (not preliminary_coarse and len(campaign['references']) != 2)):
        raise ValueError('Members and two shared references are required')
    config = campaign['config']
    grid = MACGrid(config['nx'], config['ny'], 2*np.pi, 2*np.pi)
    reused_fixed = {Path(item['path']).resolve(): item for item in campaign.get('reused_fixed_inputs', [])}
    def checked(path: Path) -> tuple[dict, dict, dict]:
        actual, manifest, values = read_member(path)
        compared = actual
        if path.resolve() in reused_fixed:
            identity = reused_fixed[path.resolve()]
            if (actual.get('fixed_step') is None or actual.get('scheme') != 'sdirk3_mrsav'
                    or actual.get('rtol_velocity') != identity['original_rtol_velocity']
                    or digest(path/'config.json') != identity['config_sha256']
                    or digest(path/'manifest.json') != identity['manifest_sha256']
                    or manifest['result_sha256'] != identity['result_sha256']):
                raise ValueError(f'Incompatible reused fixed control: {path}')
            # Fixed integration never uses the velocity adaptive tolerance.
            compared = {**actual, 'rtol_velocity': config['rtol_velocity']}
        if manifest['source'] != campaign['source'] or any(
                compared.get(k) != v for k, v in config.items() if k != 'reference_step'):
            raise ValueError(f'Incompatible source or configuration: {path}')
        if 'stage_checks_sha256' in manifest and digest(path/'stage_checks.npz') != manifest['stage_checks_sha256']:
            raise ValueError(f'Stage-check checksum mismatch: {path}')
        return actual, manifest, values
    selected_references = campaign['references'][:1] if preliminary_coarse else campaign['references']
    refs = [checked(Path(r['path'])) for r in selected_references]
    if any(manifest['status'] != 'complete' for _, manifest, _ in refs):
        raise ValueError('A shared reference failed')
    coarse, fine = refs[0][2], refs[-1][2]
    np.testing.assert_allclose(coarse['output_times'], fine['output_times'], atol=1e-10, rtol=0)
    fine_velocities = [grid.pack(u, v) for u, v in zip(fine['output_u'], fine['output_v'], strict=True)]
    ref_difference = np.asarray([grid.norm(grid.pack(u, v)-ref)/max(grid.norm(ref), 1e-14)
                                  for u, v, ref in zip(coarse['output_u'], coarse['output_v'],
                                                       fine_velocities, strict=True)])
    if preliminary_coarse:
        ref_difference[:] = np.nan  # No second-level assessment is available yet.
    directory = root/'reports/kolmogorov_adaptive/adaptive_fixed'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    report: dict[str, Any] = {'status': 'running', 'campaign': str(campaign_path.resolve()),
                              'campaign_sha256': digest(campaign_path), 'campaign_status': campaign['status'],
                              'kind': campaign['kind'], 'analysis_source': provenance(),
                              'analysis_entrypoint_sha256': digest(Path(__file__)),
                              'reused_fixed_inputs': campaign.get('reused_fixed_inputs', []),
                              'preliminary_coarse': preliminary_coarse,
                              'error_reference_step': selected_references[-1]['step'],
                              'plot_options': {'xlim': [0, config['T']], 'panel_f_legend': False,
                                               'reference_refinement_curve': False,
                                               'panel_e_quantity': 'relative velocity L2 embedded difference',
                                               'panel_e_relative_denominator': 'candidate endpoint velocity L2 norm',
                                               'panel_e_sampling': 'accepted adaptive steps only'},
                              'embedded_difference_reconstruction': {
                                  'formula': 'error_u * (atol_velocity + rtol_velocity * max(sqrt(2*K_old), sqrt(2*K_new)))',
                                  'kinetic_definition': 'K = 0.5 * inner_h(velocity, velocity)',
                                  'relative_formula': 'L2_embedded_difference / L2_trial_velocity',
                                  'zero_velocity_policy': 'zero difference over zero norm is 0; nonzero difference over zero norm is undefined',
                                  'rejected_trials': 'not reconstructable: trial kinetic energy not stored'},
                              'error_sampling': 'actual saved snapshot times; nearest natural nodes for adaptive',
                              'inputs': [m['path'] for m in campaign['members']],
                              'max_reference_refinement_difference': None if preliminary_coarse else float(ref_difference.max())}
    write_json(directory/'analysis.json', report)
    try:
        records = []
        fraction = campaign['specification']['verification_fraction']
        for member in campaign['members']:
            _, _, data = checked(Path(member['path']))
            indices = sample_indices(data['output_times'], fine['output_times'])
            relative = np.asarray([grid.norm(grid.pack(u, v)-fine_velocities[i])/
                                   max(grid.norm(fine_velocities[i]), 1e-14)
                                   for u, v, i in zip(data['output_u'], data['output_v'], indices, strict=True)])
            maximum, final = float(relative.max()), float(relative[-1])
            difference = ref_difference[indices]
            verified = (not preliminary_coarse and bool(
                difference.max() <= fraction*maximum and difference[-1] <= fraction*final))
            records.append({**member, 'max_relative_velocity_error': maximum,
                            'final_relative_velocity_error': final, 'reference_verified': verified})
            np.savez_compressed(directory/f'{member["label"]}-repeat-{member["repeat"]}-errors.npz',
                                times=data['output_times'], relative_velocity_l2=relative,
                                relative_reference_difference=difference)
        rows, representatives = [], []
        labels = ['adaptive']+[f'fixed-{h:g}' for h in campaign['specification']['fixed_steps']]
        for label in labels:
            group = [m for m in records if m['label'] == label]
            if not group:
                raise ValueError(f'Missing comparison member: {label}')
            median = float(np.median([m['integration_cpu_seconds'] for m in group]))
            chosen = min(group, key=lambda m: abs(m['integration_cpu_seconds']-median))
            representatives.append(chosen)
            rows.append({'label': label, 'fixed_step': chosen['fixed_step'],
                         'completed_repeats': sum(m['status'] == 'complete' for m in group),
                         'expected_repeats': campaign['specification']['repeats'],
                         'median_cpu_seconds': median,
                         'minimum_cpu_seconds': min(m['integration_cpu_seconds'] for m in group),
                         'maximum_cpu_seconds': max(m['integration_cpu_seconds'] for m in group),
                         'max_relative_velocity_error': max(m['max_relative_velocity_error'] for m in group),
                         'max_abs_r': max(m['max_abs_r'] for m in group),
                         'reference_verified': all(m['reference_verified'] for m in group),
                         'all_complete': all(m['status'] == 'complete' for m in group),
                         'representative_path': chosen['path']})
        plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 11})
        fig, panels = plt.subplots(2, 3, figsize=(18, 10))
        axes = panels.ravel()
        markers = ('o', '*', 'd', 's', '^', 'v', 'P', 'X', '>')
        style_steps = (None, .1, .05, .025, .001, .0005, .01, .005, .0025)
        for index, member in enumerate(representatives):
            _, _, data = checked(Path(member['path']))
            label = 'Adaptive' if member['fixed_step'] is None else rf'Fixed $\tau={member["fixed_step"]:g}$'
            if member['status'] != 'complete':
                label += f' (failed at t={member["final_time"]:.3g})'
            elif not member['reference_verified'] and not preliminary_coarse:
                label += ' (reference unverified)'
            accepted = data['attempt_accepted']
            nodes = data['times'][1:]
            h = data['attempt_h'][accepted]
            style_index = (style_steps.index(member['fixed_step'])
                           if member['fixed_step'] in style_steps else index)
            style = {'color': f'C{style_index % 10}', 'linestyle': '-' if index == 0 else '--',
                     'marker': markers[style_index % len(markers)], 'markersize': 3.,
                     'markevery': max(1, len(nodes)//18), 'linewidth': 1.1, 'label': label}
            if len(nodes):
                axes[0].step(np.r_[0., nodes], np.r_[h[0], h], where='post', **style)
                axes[1].plot(np.r_[0., nodes], np.arange(len(nodes)+1), **style)
                axes[2].plot(np.r_[0., nodes], np.r_[0., data['attempt_cpu'][accepted]], **style)
            else:
                axes[0].plot([], [], **style)
            positive = abs(data['diagnostic_r']) > 0
            axes[3].semilogy(data['times'][positive], abs(data['diagnostic_r'][positive]), **style)
            if member['fixed_step'] is None:
                embedded, scale = accepted_embedded_difference(
                    data, atol=config['atol_velocity'], rtol=config['rtol_velocity'])
                trial_norm = np.sqrt(2*data['diagnostic_kinetic'][1:])
                relative_embedded = relative_embedded_difference(embedded, trial_norm)
                positive = relative_embedded > 0
                axes[4].semilogy(nodes[positive], relative_embedded[positive], color='C0', linewidth=1.)
                np.savez_compressed(directory/'adaptive-embedded_difference.npz',
                                    times=nodes, velocity_l2_embedded_difference=embedded,
                                    trial_velocity_l2=trial_norm,
                                    relative_velocity_l2_embedded_difference=relative_embedded,
                                    normalization_scale=scale, normalized_error_u=data['attempt_error_u'][accepted])
                report['embedded_difference_reconstruction'].update(
                    representative_path=member['path'], accepted_steps=len(nodes))
                if rejected_replay is not None:
                    rejected = rejected_embedded_difference(rejected_replay, Path(member['path']), data)
                    if 'trial_velocity_l2' not in rejected:
                        raise ValueError('Rejected replay lacks trial velocity norms; replay with the updated entrypoint')
                    rejected_relative = relative_embedded_difference(
                        rejected['velocity_l2_embedded_difference'], rejected['trial_velocity_l2'])
                    positive = rejected_relative > 0
                    axes[4].scatter(rejected['trial_time'][positive],
                                    rejected_relative[positive],
                                    marker='x', color='C3', s=25, linewidths=1., zorder=4,
                                    label='Rejected trials')
                    axes[4].legend(fontsize=9)
                    shutil.copyfile(rejected_replay/'rejected_difference.npz',
                                    directory/'rejected_difference.npz')
                    np.savez_compressed(directory/'rejected-relative_embedded_difference.npz',
                                        times=rejected['trial_time'],
                                        relative_velocity_l2_embedded_difference=rejected_relative)
                    shutil.copyfile(rejected_replay/'manifest.json', directory/'rejected_replay_manifest.json')
                    report['plot_options']['panel_e_sampling'] = 'accepted and rejected adaptive trials'
                    report['embedded_difference_reconstruction']['rejected_trials'] = {
                        'replay': str(rejected_replay.resolve()),
                        'manifest_sha256': digest(rejected_replay/'manifest.json'),
                        'count': len(rejected['trial_time']), 'time_coordinate': 'attempt_t + attempt_h'}
            with np.load(directory/f'{member["label"]}-repeat-{member["repeat"]}-errors.npz') as error:
                positive = error['relative_velocity_l2'] > 0
                axes[5].semilogy(error['times'][positive], error['relative_velocity_l2'][positive], **style)
        axes[0].set_yscale('log')
        ylabels = ('Step sizes', 'Accepted-step count', 'Integration CPU time (s)',
                   r'$|r|$', r'Relative $L^2$ embedded difference', r'Relative velocity $L^2$ error')
        for index, axis in enumerate(axes):
            axis.set(xlabel=r'$t$', ylabel=ylabels[index], xlim=(0, config['T']))
            axis.grid(True, alpha=.3, linestyle='--', linewidth=.6)
            axis.text(.5, -.25, f'({chr(97+index)})', transform=axis.transAxes, ha='center')
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='lower center', ncol=3, fontsize=10,
                   bbox_to_anchor=(.5, .005))
        fig.subplots_adjust(left=.07, right=.97, top=.93,
                            bottom=.2 if len(labels) > 6 else .17, wspace=.35, hspace=.52)
        fig.savefig(directory/'six_panel.png', dpi=180, bbox_inches='tight')
        fig.savefig(directory/'six_panel.pdf', bbox_inches='tight')
        plt.close(fig)
        with (directory/'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        unresolved = any(not row['reference_verified'] for row in rows if row['all_complete'])
        report.update(status=('preliminary_reference_unverified' if preliminary_coarse else
                              'complete_with_unverified_reference' if unresolved else 'complete'),
                      records=records, summary=rows, figures=['six_panel.png', 'six_panel.pdf'],
                      has_failed_controls=any(not row['all_complete'] for row in rows))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(directory/'analysis.json', report)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--rejected-replay', type=Path)
    parser.add_argument('--output-root', type=Path, default=PROJECT)
    parser.add_argument('--preliminary-coarse', action='store_true')
    args = parser.parse_args()
    print(analyze(args.campaign, root=args.output_root, rejected_replay=args.rejected_replay,
                  preliminary_coarse=args.preliminary_coarse))

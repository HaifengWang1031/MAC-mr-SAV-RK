"""Read saved joint-tolerance scans; compute reference checks and Pareto choices."""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.kolmogorov_adaptive.analyze import aligned_output_times, read_member
from experiments.records import PROJECT, digest, provenance, write_json
from solver.mac.grid import MACGrid


def pareto_rows(rows: list[dict]) -> list[dict]:
    """Minimize CPU, observed velocity error, and scalar error simultaneously."""
    keys = ('median_cpu_seconds', 'max_relative_velocity_error', 'max_abs_r')
    candidates = [row for row in rows if row['eligible'] and row['atol_r'] is not None]
    return [row for row in candidates if not any(
        all(other[key] <= row[key] for key in keys) and
        any(other[key] < row[key] for key in keys)
        for other in candidates)]


def summarize(records: list[dict], cases: list[dict], repeats: int) -> list[dict]:
    rows = []
    for case in cases:
        group = [r for r in records if r['case_id'] == case['id']]
        complete = [r for r in group if r['status'] == 'complete']
        row = {**case, 'completed_repeats': len(complete), 'expected_repeats': repeats,
               'eligible': len(complete) == repeats and
               {r['repeat'] for r in complete} == set(range(repeats)) and
               all(r['reference_verified'] for r in complete)}
        for target, key, operation in (
            ('median_cpu_seconds', 'integration_cpu_seconds', np.median),
            ('min_cpu_seconds', 'integration_cpu_seconds', np.min),
            ('max_cpu_seconds', 'integration_cpu_seconds', np.max),
            ('max_relative_velocity_error', 'max_relative_velocity_error', np.max),
            ('final_relative_velocity_error', 'final_relative_velocity_error', np.max),
            ('max_abs_r', 'max_abs_r', np.max),
            ('accepted_steps', 'accepted_steps', np.median),
            ('rejected_trials', 'rejected_trials', np.median),
            ('max_step_fraction', 'max_step_fraction', np.median),
        ):
            row[target] = float(operation([r[key] for r in complete])) if complete else None
        row['reference_verified'] = bool(complete) and all(r['reference_verified'] for r in complete)
        rows.append(row)
    return rows


def analyze_scan(campaign_path: Path, *, root: Path = PROJECT) -> Path:
    campaign = json.loads(campaign_path.read_text())
    if len(campaign['references']) != 2:
        raise ValueError('Two completed reference trajectories are required')
    config = campaign['base_config']
    grid = MACGrid(config['nx'], config['ny'], 2*np.pi, 2*np.pi)

    def checked_data(path: Path) -> dict:
        actual, manifest, data = read_member(path)
        if manifest['status'] != 'complete' or manifest['source'] != campaign['source']:
            raise ValueError(f'Failed or mixed-source member: {path}')
        variable_keys = {'atol_velocity', 'rtol_velocity', 'reference_step'}
        if any(actual.get(k) != value for k, value in config.items() if k not in variable_keys):
            raise ValueError(f'Member has incompatible physical/numerical inputs: {path}')
        aligned_output_times(config, data, str(path))
        return data

    coarse, fine = [checked_data(Path(r['path'])) for r in campaign['references']]
    fine_velocities = [grid.pack(u, v) for u, v in zip(fine['output_u'], fine['output_v'], strict=True)]

    def errors(data: dict) -> np.ndarray:
        return np.asarray([grid.norm(grid.pack(u, v)-ref)/max(grid.norm(ref), 1e-14)
                           for u, v, ref in zip(data['output_u'], data['output_v'],
                                                fine_velocities, strict=True)])

    reference_difference = errors(coarse)
    folder = root/'reports/kolmogorov_adaptive/joint_tolerance_scan'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    folder.mkdir(parents=True)
    report: dict[str, Any] = {'status': 'running', 'campaign': str(campaign_path.resolve()),
                              'campaign_status': campaign['status'],
                              'analysis_source': provenance(),
                              'campaign_sha256': digest(campaign_path), 'kind': campaign['kind'],
                              'T': config['T'], 'output_every': config['output_every'],
                              'error_sampling': 'exact uniform output nodes; scalar uses all accepted nodes',
                              'max_reference_difference': float(reference_difference.max()),
                              'final_reference_difference': float(reference_difference[-1])}
    write_json(folder/'analysis.json', report)
    try:
        records = []
        fraction = campaign['specification']['verification_fraction']
        cases = {case['id']: case for case in campaign['cases']}
        for member in campaign['members']:
            if member['status'] != 'complete':
                records.append({**member, 'reference_verified': False})
                continue
            path = Path(member['path'])
            actual = json.loads((path/'config.json').read_text())
            case = cases[member['case_id']]
            if (actual['atol_velocity'], actual['rtol_velocity'], actual['atol_scalar']) != (
                    case['atol_u'], case['rtol_u'], case['atol_r']):
                raise ValueError(f'Member tolerance mismatch: {path}')
            velocity_error = errors(checked_data(path))
            maximum, final = float(velocity_error.max()), float(velocity_error[-1])
            verified = reference_difference.max() <= fraction*maximum and \
                reference_difference[-1] <= fraction*final
            records.append({**member, 'max_relative_velocity_error': maximum,
                            'final_relative_velocity_error': final, 'reference_verified': bool(verified)})
            np.savez_compressed(folder/f'{member["case_id"]}-repeat-{member["repeat"]}-errors.npz',
                                times=fine['output_times'], relative_velocity_l2=velocity_error,
                                relative_reference_difference=reference_difference)
        rows = summarize(records, campaign['cases'], campaign['specification']['repeats'])
        pareto = pareto_rows(rows)
        recommendations = []
        for velocity_target, scalar_target in product(
                campaign['specification']['velocity_error_targets'],
                campaign['specification']['scalar_error_targets']):
            candidates = [row for row in rows if row['eligible'] and row['atol_r'] is not None
                          and row['max_relative_velocity_error'] <= velocity_target
                          and row['max_abs_r'] <= scalar_target]
            best = min(candidates, key=lambda row: row['median_cpu_seconds']) if candidates else None
            recommendations.append({'velocity_error_target': velocity_target,
                                    'scalar_error_target': scalar_target, 'choice': best})
        report.update(status='complete', records=records, summary=rows,
                      pareto=pareto, recommendations=recommendations,
                      recommendation_scope=('pilot only; long-time peak is outside this window'
                                            if campaign['kind'] == 'pilot' else 'this grid, horizon and scanned candidates'))
        with (folder/'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        usable = [r for r in rows if r['median_cpu_seconds'] is not None]
        labeled = {choice['choice']['id'] for choice in recommendations if choice['choice'] is not None}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for row in usable:
            for axis, key in zip(axes, ('max_relative_velocity_error', 'max_abs_r')):
                if row[key] > 0:
                    axis.loglog(row['median_cpu_seconds'], row[key],
                                's' if row['role'] == 'baseline' else 'o',
                                color='C0' if row['eligible'] else '0.65',
                                label='Velocity-only baseline' if row['role'] == 'baseline' else None)
                    if row['id'] in labeled:
                        axis.annotate(row['id'], (row['median_cpu_seconds'], row[key]),
                                      xytext=(3, 3), textcoords='offset points', fontsize=7)
        for axis, ylabel in zip(axes, ('Maximum relative velocity error at output nodes', 'Maximum |r|')):
            axis.set(xlabel='Median integration CPU time (s)', ylabel=ylabel)
            axis.grid(True, alpha=.25)
            if any(row['role'] == 'baseline' for row in usable):
                axis.legend(fontsize=8)
        fig.suptitle(f'SDIRK3-mrSAV I, T={config["T"]:g}, {campaign["kind"]}')
        fig.savefig(folder/'accuracy_cost.png', dpi=180)
        fig.savefig(folder/'accuracy_cost.pdf')
        plt.close(fig)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(folder/'analysis.json', report)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(analyze_scan(args.campaign, root=args.root))

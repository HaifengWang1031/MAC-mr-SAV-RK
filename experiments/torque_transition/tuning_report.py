"""Read-only temporal qualification/ranking for explicit MAC adaptive-search plans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.records import digest, write_json


def load_job(directory: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Validate recorded payloads and metadata before any error or timing comparison."""
    row = json.loads((directory/'result.json').read_text())
    if digest(directory/'config.json') != row['config_sha256']:
        raise ValueError('Configuration checksum mismatch')
    for filename, checksum in row['artifacts'].items():
        if digest(directory/filename) != checksum:
            raise ValueError(f'Payload checksum mismatch: {directory/filename}')
    with np.load(directory/'final.npz') as saved:
        if str(saved['discretization']) != 'mac' or int(saved['nodes']) != row['grid_cells'] \
                or int(saved['case']) != row['case'] or not np.isclose(
                    float(saved['t']), row['final_t'], rtol=0., atol=1e-10):
            raise ValueError('Field metadata mismatch')
        velocity = saved['velocity'].copy()
    return row, velocity


def qualification(candidate: dict[str, Any], error: float, refinement: float,
                  observable_errors: dict[str, float], observable_refinements: dict[str, float],
                  gate: dict[str, float]) -> tuple[bool, list[str]]:
    """Keep accuracy, reference resolution, physical gates and cost strictly separate."""
    reasons = []
    if candidate['status'] != 'complete':
        reasons.append('incomplete')
    if not np.isfinite([error, refinement, *observable_errors.values(),
                        *observable_refinements.values(), candidate['max_coupling_deviation'],
                        candidate['max_full_divergence'], candidate['max_stage_residual'],
                        candidate['max_error_u'], candidate['max_error_r']]).all():
        reasons.append('nonfinite')
    if error+refinement > gate['velocity_relative']:
        reasons.append('velocity_budget')
    if refinement > gate['reference_fraction']*gate['velocity_relative'] or \
            refinement > gate['reference_fraction']*max(error, 1e-12):
        reasons.append('reference_unresolved')
    if any(observable_errors[key]+observable_refinements[key] > gate['observables_relative']
           for key in observable_errors):
        reasons.append('observable_budget')
    if any(value > gate['reference_fraction']*gate['observables_relative']
           for value in observable_refinements.values()):
        reasons.append('observable_reference_unresolved')
    if candidate['max_coupling_deviation'] > gate['coupling']:
        reasons.append('sav_coupling')
    if candidate['max_full_divergence'] > gate['full_divergence']:
        reasons.append('full_divergence')
    if candidate['max_stage_residual'] > 1e-8 or max(candidate['max_error_u'], candidate['max_error_r']) > 1+1e-12:
        reasons.append('stage_or_acceptance')
    return not reasons, reasons


def analyze(batch: Path, output: Path) -> dict[str, Any]:
    """Compare common endpoint fields; the reader never imports an integrator or model."""
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((batch/'plan.json').read_text())
    execution = json.loads((batch/'execution.json').read_text())
    if execution['status'] != 'complete' or execution['plan_sha256'] != digest(batch/'plan.json'):
        raise ValueError('Execution incomplete or plan changed')
    coarse, uc = load_job(batch/'jobs/reference')
    fine, uf = load_job(batch/'jobs/reference-half')
    if coarse['status'] != 'complete' or fine['status'] != 'complete':
        raise ValueError('Reference failed')
    identity_keys = ('grid_cells', 'case', 'seed_sha256', 'start_t', 'duration')
    if any(coarse[key] != fine[key] for key in identity_keys) or not np.isclose(
            coarse['final_t'], fine['final_t'], rtol=0., atol=1e-10):
        raise ValueError('Coarse/fine reference windows differ')
    if coarse['source']['code_sha256'] != fine['source']['code_sha256']:
        raise ValueError('Reference source identities differ')
    if uc.shape != uf.shape or not np.isfinite(uc).all() or not np.isfinite(uf).all():
        raise ValueError('Invalid reference fields')
    norm = float(np.linalg.norm(uf))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Reference norm must be positive')
    refinement = float(np.linalg.norm(uc-uf)/norm)
    keys = ('energy', 'angular_momentum', 'power', 'dissipation')
    scales = {key: max(abs(fine['observables'][key]), 1e-14) for key in keys}
    scales['angular_momentum'] = max(scales['angular_momentum'],
        np.sqrt(2*fine['observables']['energy'])*2*np.pi*1e-3)
    obs_ref = {key: abs(coarse['observables'][key]-fine['observables'][key])/scales[key] for key in keys}
    rows = []
    inputs = {str(batch/'plan.json'): digest(batch/'plan.json')}
    for job in plan['jobs']:
        path = batch/'jobs'/job['name']
        row, velocity = load_job(path)
        if json.loads((path/'config.json').read_text()) != job:
            raise ValueError('Executed configuration differs from planned job')
        if row['source']['code_sha256'] != fine['source']['code_sha256']:
            raise ValueError('Source identity differs between candidate and reference')
        inputs[str(path/'result.json')] = digest(path/'result.json')
        if any(row[key] != fine[key] for key in ('grid_cells', 'case', 'seed_sha256', 'start_t', 'duration')):
            raise ValueError('Comparison requires identical physical window and seed')
        aligned = np.isclose(row['final_t'], fine['final_t'], rtol=0., atol=1e-10)
        if row['status'] == 'complete' and not aligned:
            raise ValueError('Complete endpoint differs from reference')
        if job['mode'] == 'reference':
            continue
        error: float | None = None
        obs_error: dict[str, float] | None = None
        if row['status'] != 'complete':
            eligible, reasons = False, ['incomplete']
        else:
            error = float(np.linalg.norm(velocity-uf)/norm)
            obs_error = {key: abs(row['observables'][key]-fine['observables'][key])/scales[key] for key in keys}
            eligible, reasons = qualification(row, error, refinement, obs_error, obs_ref, plan['gate'])
        rows.append({'name': row['name'], 'settings': row['settings'], 'eligible': eligible,
                     'reasons': reasons, 'velocity_error': error, 'observable_errors': obs_error,
                     **{key: row[key] for key in ('wall_seconds', 'cpu_seconds', 'accepted_steps',
                                                 'attempts', 'scalar_dominant_fraction',
                                                 'max_coupling_deviation', 'max_full_divergence')}})
    groups = []
    for name in dict.fromkeys(row['settings']['name'] for row in rows):
        selected = [row for row in rows if row['settings']['name'] == name]
        groups.append({'name': name, 'eligible': all(row['eligible'] for row in selected),
                       'settings': selected[0]['settings'], 'repeats': len(selected),
                       'median_wall_seconds': float(np.median([row['wall_seconds'] for row in selected])),
                       'median_cpu_seconds': float(np.median([row['cpu_seconds'] for row in selected])),
                       'max_velocity_error': max((row['velocity_error'] for row in selected
                                                  if row['velocity_error'] is not None), default=None)})
    ranked = sorted([group for group in groups if group['eligible']], key=lambda r: r['median_wall_seconds'])
    result = {'status': 'complete' if ranked else 'no_qualified_candidate',
              'analysis_source_sha256': digest(Path(__file__)), 'inputs': inputs,
              'reference_refinement': refinement, 'observable_refinements': obs_ref,
              'gate': plan['gate'], 'rows': rows, 'ranking': ranked,
              'winner': ranked[0] if ranked else None, 'workers': execution['workers'],
              'physical_window': {'case': fine['case'], 'start': fine['start_t'], 'end': fine['final_t']},
              'limits': 'Endpoint temporal comparison on one MAC seed; not stationary, grid or long-time qualification.',
              'production_T': 100000., 'production_approved': False}
    write_json(output/'analysis.json', result)
    lines = ['# MAC256 adaptive parameter search', '', '**Production T=100000 not approved or started.**', '',
             f"Window: X{fine['case']}, {fine['start_t']} → {fine['final_t']}; workers={execution['workers']}.",
             f'Reference refinement relative L2: {refinement:.6e}.', '',
             '| Candidate | Eligible | Steps / trials | Scalar-dominant | Velocity error | Wall / s | Reasons |',
             '|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        dominance = row['scalar_dominant_fraction']
        dominance_text = 'n/a' if dominance is None else f'{dominance:.1%}'
        error_text = 'n/a' if row['velocity_error'] is None else f"{row['velocity_error']:.3e}"
        lines.append(f"| {row['name']} | {row['eligible']} | {row['accepted_steps']}/{row['attempts']} | "
                     f"{dominance_text} | {error_text} | {row['wall_seconds']:.3f} | "
                     f"{', '.join(row['reasons'])} |")
    lines += ['', '## Qualified ranking', '']
    for row in ranked:
        lines.append(f"- {row['name']}: median {row['median_wall_seconds']:.3f}s, "
                     f"max relative velocity error {row['max_velocity_error']:.3e}, repeats={row['repeats']}.")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.batch, args.output)

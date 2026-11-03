"""Readonly interleaved adaptive timings and common-schedule numerical comparisons."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from skfem import MeshTri

from experiments.records import PROJECT, provenance, write_json
from solver.fem.space import MeshSpace
from .plots import plot_performance
from .records import fingerprints, load_jobs
from .reconstruct import decode_velocity

COSTS = ('setup_cpu_seconds', 'integration_cpu_seconds', 'observer_cpu_seconds',
         'total_body_cpu_seconds', 'total_body_wall_seconds', 'process_wall_seconds', 'peak_rss_bytes')


def analyze(directory: Path, *, root: Path = PROJECT) -> Path:
    directory, root = Path(directory).resolve(), Path(root).resolve()
    report = root/'reports/fem_adaptive_benchmark'/directory.name
    if report.is_relative_to(directory):
        raise ValueError('Performance reports must be separate from raw campaigns')
    before = fingerprints(directory)
    manifest, jobs = load_jobs(directory)
    if manifest['experiment'] != 'fem_adaptive_benchmark' or manifest['repeats'] < 3:
        raise ValueError('Performance conclusions require at least three interleaved repeats')
    identities = {label: {item['result']['source']['code_sha256'] for item in jobs if item['job']['label'] == label}
                  for label in ('baseline', 'direct', 'recycled')}
    environments = {json.dumps({key: value for key, value in item['result']['source'].items()
                                if key != 'code_sha256'}, sort_keys=True) for item in jobs}
    if any(len(values) != 1 for values in identities.values()) or identities['direct'] != identities['recycled'] \
            or len(environments) != 1:
        raise ValueError('Performance repeats must use stable sources and an identical environment')
    cfg = json.loads((directory/'input/config.json').read_text())
    with np.load(directory/'input/native.npz', allow_pickle=False) as native:
        basis, _ = MeshSpace(MeshTri(native['coordinates'], native['connectivity'], sort_t=False),
                            cfg['quadrature_order']).bases()
    table: dict[str, Any] = {}
    for mode in ('adaptive', 'replay'):
        table[mode] = {}
        for label in ('baseline', 'direct', 'recycled'):
            selected = [item for item in jobs if item['job']['mode'] == mode and item['job']['label'] == label
                        and not item['job']['profiled']]
            if sorted(item['job']['repeat'] for item in selected) != list(range(manifest['repeats'])):
                raise ValueError('Missing or duplicate unprofiled performance repeat')
            costs: dict[str, list[float]] = {key: [] for key in COSTS}
            for item in selected:
                result, job = item['result'], item['job']
                for key in COSTS:
                    value = result['metrics'].get(key, result.get(key, job.get(key)))
                    if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                        raise ValueError('Invalid worker resource metric')
                    costs[key].append(float(value))
            table[mode][label] = {key: {'median': float(np.median(values)), 'min': min(values), 'max': max(values)}
                                  for key, values in costs.items()}
            table[mode][label]['accepted_steps'] = [item['result']['metrics']['accepted_steps'] for item in selected]
            table[mode][label]['rejected_steps'] = [item['result']['metrics']['rejected_steps'] for item in selected]
            table[mode][label]['counters'] = {key: [item['result']['metrics'].get(key) for item in selected] for key in (
                'stokes_factorizations', 'stokes_recycled_solves', 'stokes_krylov_iterations',
                'stokes_preconditioner_solves', 'stokes_fallbacks', 'stokes_max_coefficient_residual')}
    reference = next(item['fields'] for item in jobs if item['job']['mode'] == 'replay'
                     and item['job']['label'] == 'direct' and item['job']['repeat'] == 0)
    full_reference = decode_velocity(basis, reference['u'], reference['v'])
    quadrature_reference = np.asarray(basis.interpolate(full_reference))
    denominator = np.sqrt(np.sum(quadrature_reference**2*basis.dx))
    comparisons = []
    for item in jobs:
        if item['job']['mode'] != 'replay':
            continue
        fields = item['fields']
        if not np.array_equal(fields['times'], reference['times']):
            raise ValueError('Replay did not use a common accepted schedule')
        difference = basis.interpolate(decode_velocity(basis, fields['u'], fields['v'])-full_reference)
        relative = float(np.sqrt(np.sum(np.asarray(difference)**2*basis.dx))/denominator)
        scalar = float(abs(fields['r']-reference['r']))
        force = max(float(np.max(abs(fields[key]-reference[key]))) for key in ('drag', 'lift', 'pressure'))
        comparisons.append({'label': item['job']['label'], 'repeat': item['job']['repeat'],
                            'relative_velocity_L2': relative, 'scalar_absolute_difference': scalar,
                            'max_force_pressure_absolute_difference': force,
                            'passed': relative <= 1e-8 and scalar <= 1e-8 and force <= 1e-7})
    direct_history = next(item['fields'] for item in jobs if item['job']['mode'] == 'adaptive'
                          and item['job']['label'] == 'direct' and item['job']['repeat'] == 0)
    decisions_match = all(np.array_equal(item['fields']['accepted'], direct_history['accepted'])
                          for item in jobs if item['job']['mode'] == 'adaptive')
    chosen = min(('direct', 'recycled'), key=lambda label: table['adaptive'][label]['total_body_cpu_seconds']['median'])
    gains = {mode: {label: {key: table[mode]['baseline'][key]['median']/table[mode][label][key]['median']
                            for key in COSTS if key != 'peak_rss_bytes'} for label in ('direct', 'recycled')}
             for mode in ('adaptive', 'replay')}
    summary = {'run_id': directory.name, 'seed_run_id': manifest['seed_run_id'], 'duration': manifest['duration'],
               'developed_seed': manifest['seed_cycles']['developed'], 'triangles': basis.mesh.t.shape[1],
               'source_change_verified': identities['baseline'] != identities['direct'],
               'repeats': manifest['repeats'], 'table': table, 'speedups': gains, 'selected_backend': chosen,
               'common_schedule_comparisons': comparisons, 'adaptive_decisions_match': decisions_match,
               'numerical_comparison_passed': decisions_match and all(item['passed'] for item in comparisons),
               'worker_sources': {item['job']['label']: item['result']['source'] for item in jobs},
               'analysis_source': provenance(), 'input_fingerprints': before,
               'limits': 'Single-thread repeated/interleaved native developed-state window, not an end-to-end '
                         'speed guarantee for every mesh or controller. Profiled runs are excluded from timings. '
                         'Body costs include setup and physical-pressure observations; process costs also include imports. '
                         'No controller, equation, tolerance or accepted schedule was changed for the replay.'}
    figures = report/'figures'; figures.mkdir(parents=True, exist_ok=True)
    plot_performance(summary, figures)
    write_json(report/'analysis.json', summary)
    with (report/'timings.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['mode', 'backend', 'quantity', 'median', 'min', 'max'])
        for mode, backends in table.items():
            for label, values in backends.items():
                for key in COSTS: writer.writerow([mode, label, key, *[values[key][part] for part in ('median', 'min', 'max')]])
    if fingerprints(directory) != before:
        raise RuntimeError('Raw performance inputs changed during readonly analysis')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(analyze(args.run, root=args.root))


if __name__ == '__main__': main()

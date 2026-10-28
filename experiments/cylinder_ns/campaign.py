"""Readonly developed-flow qualification from explicit mesh/geometry/time reference IDs."""
import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from experiments.workflow import PROJECT, provenance, write_json
from .benchmark import REFERENCE, cycle_statistics, reference_comparison
from .performance import analyze as analyze_performance
from .plots import animate, plot_cycles, plot_fields, plot_mesh, plot_refinement, plot_reference_steps
from .records import fingerprints, load_reference


def cycle_sensitivity(first: dict, second: dict, *, tolerance: float) -> dict:
    """Compare the same seven nonzero primary observables as the reference gate."""
    if not all(key in first and key in second for key in REFERENCE):
        return {'passed': False, 'reason': 'Missing three resolved cycles', 'tolerance': tolerance}
    errors = {key: abs(first[key]-second[key])/max(abs(second[key]), 1e-12*abs(REFERENCE[key])) for key in REFERENCE}
    return {'passed': all(value <= tolerance for value in errors.values()),
            'relative_differences': errors, 'tolerance': tolerance,
            'scope': 'CD/CL extrema, lift Strouhal, pressure extrema; means/midranges/amplitudes are additional diagnostics'}


def analyze(coarse: Path, fine: Path, tight: Path, geometry: Path, *,
            performance: Path | None = None, root: Path = PROJECT) -> Path:
    paths = [Path(p).resolve() for p in (coarse, fine, tight, geometry)]
    if len(set(paths)) != 4:
        raise ValueError('Four explicit, distinct refinement records are required')
    before = {str(p): fingerprints(p) for p in paths}
    records = [load_reference(p) for p in paths]
    for label, record in zip(('coarse', 'fine', 'tight', 'geometry'), records):
        fields, cfg = record['fields'], record['config']
        record.update(label=label, cycles=cycle_statistics(fields['observation_time'], fields['drag'],
                      fields['lift'], fields['pressure'], mean_velocity=2*cfg['inlet_max']/3, diameter=2*cfg['radius']))
    a, b, t, g = records
    physics = ('nu', 'inlet_max', 'lx', 'ly', 'radius', 'scheme', 'gamma', 'convection', 'forcing',
               'pressure_semantics', 'physical_pressure_semantics')
    if any(any(record['config'][key] != b['config'][key] for key in physics) for record in records):
        raise ValueError('Refinement records do not share physical equations/time format')
    if not (a['config']['resolution'] < b['config']['resolution'] and
            a['basis'].mesh.t.shape[1] < b['basis'].mesh.t.shape[1]):
        raise ValueError('Spatial comparison requires an actually finer bulk mesh')
    if g['config']['resolution'] != b['config']['resolution'] \
            or g['config']['circle_segments'] <= b['config']['circle_segments']:
        raise ValueError('Geometry comparison must hold bulk resolution fixed and refine the circle')
    for key in ('coordinates', 'connectivity', 'component_u', 'component_v', 'u', 'v', 'r', 'time'):
        if not np.array_equal(b['native'][key], t['native'][key]):
            raise ValueError('Temporal comparison must use identical native mesh and initial coefficients')
    steps = [record['result']['step'] for record in records]
    if all(step is not None for step in steps):
        if steps[0] != steps[1] or steps[3] != steps[1] or not steps[2] <= .5*steps[1]:
            raise ValueError('Isolate space/geometry at one step and halve the temporal step')
        temporal_control = 'fixed step halved'
    elif all(step is None for step in steps):
        for record in (a, g):
            if any(record['config'][key] != b['config'][key] for key in ('controller', 'scalar_atol')):
                raise ValueError('Adaptive spatial/geometry comparisons must keep identical error controls')
        first, second = b['config']['controller'], t['config']['controller']
        if not (second['rtol'] < first['rtol'] and second['atol'] <= first['atol'] \
                and second['max_step'] <= first['max_step']):
            raise ValueError('Adaptive temporal reference must tighten velocity tolerance and not enlarge max_step')
        if b['config']['scalar_atol'] is not None and not 0. < t['config']['scalar_atol'] < b['config']['scalar_atol']:
            raise ValueError('Adaptive temporal reference must also tighten active scalar control')
        temporal_control = 'velocity/scalar tolerances tightened on identical native state and mesh'
    else:
        raise ValueError('Do not mix fixed/adaptive controls in isolated refinement comparisons')
    spatial = cycle_sensitivity(a['cycles'], b['cycles'], tolerance=.02)
    temporal = cycle_sensitivity(b['cycles'], t['cycles'], tolerance=.01)
    geometric = cycle_sensitivity(b['cycles'], g['cycles'], tolerance=.02)
    comparisons = {record['label']: reference_comparison(record['cycles'], record['config'], complete=True)
                   for record in records}
    developed = all(record['cycles']['developed'] for record in records)
    qualified = developed and comparisons['tight']['passed'] and comparisons['geometry']['passed'] \
        and spatial['passed'] and temporal['passed'] and geometric['passed']
    identifier = fine.name+'-'+hashlib.sha256(repr(before).encode()).hexdigest()[:8]
    report = Path(root).resolve()/'reports/cylinder_campaign'/identifier
    if any(report.is_relative_to(path) for path in paths):
        raise ValueError('Reports must be separate from raw refinement records')
    figures = report/'figures'; figures.mkdir(parents=True, exist_ok=True)
    plot_mesh(t['basis'].mesh, figures)
    display = plot_fields(t['basis'], t['fields']['u'], t['fields']['v'],
                          f"{t['config']['scheme']}, Re={t['config']['reynolds']:g}, "
                          f"t={t['result']['final_time']:.5f}; saved tight reference", figures)
    plot_reference_steps(t['fields'], figures)
    movie = animate(t['basis'], t['fields']['snapshot_u'], t['fields']['snapshot_v'],
                    t['fields']['snapshot_times'], figures)
    plot_cycles({'time': t['fields']['observation_time'], 'drag': t['fields']['drag'],
                 'lift': t['fields']['lift'], 'pressure_difference': t['fields']['pressure']}, t['cycles'], figures)
    if all(all(key in record['cycles'] for key in REFERENCE) for record in records):
        plot_refinement(records, figures)
    performance_report = analyze_performance(performance, root=root) if performance is not None else None
    summaries: list[dict[str, Any]] = []
    for record in records:
        summaries.append({'label': record['label'], 'run_id': record['directory'].name,
                          'path': str(record['directory']), 'seed_run_id': record['manifest']['seed_run_id'],
                          'resolution': record['config']['resolution'], 'circle_segments': record['config']['circle_segments'],
                          'triangles': record['basis'].mesh.t.shape[1], 'fixed_step': record['result']['step'],
                          'initial_time': record['result']['initial_time'], 'final_time': record['result']['final_time'],
                          'initial_r': float(record['native']['r']), 'final_r': float(record['fields']['r']),
                          'controller': record['config']['controller'], 'scalar_atol': record['config']['scalar_atol'],
                          'cycles': record['cycles'], 'reference': comparisons[record['label']],
                          'metrics': record['result']['metrics'], 'producer': record['result']['source']})
    summary = {'qualified': bool(qualified), 'developed_all': developed, 'records': summaries,
               'spatial_sensitivity': spatial, 'temporal_sensitivity': temporal, 'geometry_sensitivity': geometric,
               'temporal_control': temporal_control,
               'performance_report': None if performance_report is None else str(performance_report),
               'analysis_source': provenance(), 'input_fingerprints': before, 'display': display, 'animation': movie,
               'pressure_semantics': 'separate instantaneous constrained-rate pressure, not incremental stage multipliers',
               'limits': 'Native restarts interpolate a named developed seed (zero extension inside its polygon) '
                         'then perform an affine weak Leray projection during input preparation, not analysis. '
                         'Each refinement is independently required to redevelop its final three complete lift cycles. '
                         'Polygonal affine geometry and weak, not pointwise, incompressibility. '
                         'Reference and sensitivity gates apply to the seven primary observables, not near-zero mean lift. '
                         'Time means, midranges and peak-to-peak values remain separately reported diagnostics. '
                         'No PDE order proof or certified reference interval is claimed.'}
    write_json(report/'analysis.json', summary)
    with (report/'cycles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); keys = ['period', *REFERENCE, *[name+'_'+part for name in ('drag','lift','pressure')
                                                                  for part in ('mean','midrange','peak_to_peak')]]
        writer.writerow(['label', 'run_id', *keys])
        for record in records:
            writer.writerow([record['label'], record['directory'].name, *[record['cycles'].get(key) for key in keys]])
    if before != {str(p): fingerprints(p) for p in paths}:
        raise RuntimeError('Refinement inputs changed during readonly reporting')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('coarse', 'fine', 'tight', 'geometry'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--performance', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(analyze(args.coarse, args.fine, args.tight, args.geometry, performance=args.performance, root=args.root))


if __name__ == '__main__': main()

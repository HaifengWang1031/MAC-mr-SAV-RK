"""Readonly fields/history/animation from an explicitly identified cylinder run."""
import argparse
import csv
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np

from experiments.fem_ns.attempts import FIELDS as ATTEMPT_FIELDS, STRINGS
from experiments.workflow import PROJECT, digest, provenance, write_json
from .observations import FIELDS as OBSERVATION_FIELDS
from .plots import animate, plot_fields, plot_histories, plot_mesh, plot_cycles, plot_reference_steps
from .benchmark import cycle_statistics, reference_comparison
from .representation import validate
from .run import load_cylinder
from .records import fingerprints, load_reference


def analyze_reference(directory: Path, *, root: Path = PROJECT) -> Path:
    """Render an explicit native-state reference, without a flow model or pressure solve."""
    directory, root = Path(directory).resolve(), Path(root).resolve()
    report = root/'reports/cylinder_reference'/directory.name
    if report.is_relative_to(directory):
        raise ValueError('Reference reports must be separate from raw runs')
    before = fingerprints(directory)
    record = load_reference(directory)
    fields, cfg, basis = record['fields'], record['config'], record['basis']
    cycles = cycle_statistics(fields['observation_time'], fields['drag'], fields['lift'], fields['pressure'],
                              mean_velocity=2*cfg['inlet_max']/3, diameter=2*cfg['radius'])
    figures = report/'figures'; figures.mkdir(parents=True, exist_ok=True)
    plot_mesh(basis.mesh, figures)
    display = plot_fields(basis, fields['u'], fields['v'],
                           f"{cfg['scheme']}, Re={cfg['reynolds']:g}, t={record['result']['final_time']:.5f}", figures)
    plot_reference_steps(fields, figures)
    plot_cycles({'time': fields['observation_time'], 'drag': fields['drag'], 'lift': fields['lift'],
                 'pressure_difference': fields['pressure']}, cycles, figures)
    movie = animate(basis, fields['snapshot_u'], fields['snapshot_v'], fields['snapshot_times'], figures)
    write_json(report/'analysis.json', {'run_id': directory.name, 'run': str(directory),
        'input_fingerprints': before, 'analysis_source': provenance(), 'producer': record['result']['source'],
        'metrics': record['result']['metrics'], 'developed_cycles': cycles,
        'reference_comparison': reference_comparison(cycles, cfg, complete=True),
        'display': display, 'animation': movie,
        'physical_pressure_coefficients_available': 'physical_pressure' in fields,
        'qualification_note': 'Reference/developed-cycle gate only; independent spatial, geometry and temporal '
                              'sensitivity evidence is still required. No computation is launched by this report.'})
    if fingerprints(directory) != before:
        raise RuntimeError('Native reference changed during readonly reporting')
    return report


def analyze(directory: Path, *, root: Path = PROJECT) -> Path:
    directory, root = Path(directory).resolve(), Path(root).resolve()
    if not (directory/'results.h5').is_file():
        return analyze_reference(directory, root=root)
    cfg, manifest = load_cylinder(directory)
    report = (root/'reports'/'cylinder_ns'/directory.name).resolve()
    if report.is_relative_to(directory):
        raise ValueError('Cylinder reports must be separate from input runs')
    before = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    source = provenance()
    source['packages']['scikit-fem'] = version('scikit-fem')
    figures = report/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    with h5py.File(directory/'results.h5', 'r') as data:
        vb, _ = validate(data, cfg, manifest)
        times, steps = data['times'][:], data['dt'][:]
        diagnostics = {key: data['diagnostics/'+key][:] for key in data['diagnostics']}
        observations = {key: data['observations/'+key][:] for key in OBSERVATION_FIELDS}
        attempts = ({key: data['attempts/'+key].asstr()[:] if key in STRINGS else data['attempts/'+key][:]
                     for key in ATTEMPT_FIELDS} if 'attempts' in data else {})
        final_time = float(data['final/t'][()])
        cycles = cycle_statistics(observations['time'], observations['drag'], observations['lift'],
                                  observations['pressure_difference'], mean_velocity=2*cfg['inlet_max']/3,
                                  diameter=2*cfg['radius'])
        comparison = reference_comparison(cycles, cfg, complete=manifest['status'] == 'complete')
        plot_mesh(vb.mesh, figures)
        display = plot_fields(vb, data['final/u'][:], data['final/v'][:],
                               f"{cfg['scheme']}, Re={cfg['reynolds']:g}, t={final_time:.5f} ({manifest['status']})", figures)
        plot_histories(times, steps, diagnostics, observations, attempts, cfg['stokes_tolerance'], figures)
        plot_cycles(observations, cycles, figures)
        movie = animate(vb, data['snapshots/u'][:], data['snapshots/v'][:],
                        data['snapshots/actual_times'][:], figures)
        summary = {'run': str(directory.relative_to(root) if directory.is_relative_to(root) else directory),
                   'run_id': directory.name, 'status': manifest['status'], 'error': manifest.get('error', ''),
                   'manifest_sha256': digest(directory/'manifest.json'),
                   'results_sha256': manifest['results_sha256'], 'analysis_source': source,
                   'final_time': final_time, 'accepted_steps': len(steps),
                   'rejected_steps': int(np.sum(~attempts['accepted'])) if attempts else 0,
                   'step_min': float(np.min(steps)) if len(steps) else None,
                   'step_max': float(np.max(steps)) if len(steps) else None,
                   'metrics': manifest['metrics'], 'field_layout': cfg['field_layout'],
                   'pressure_semantics': cfg['pressure_semantics'],
                   'physical_pressure_semantics': cfg['physical_pressure_semantics'],
                   'pressure_available': bool(data.attrs['physical_pressure_available']),
                   'max_stage_residual': float(np.max(data['stages/residual'][:], initial=0.)),
                   'max_scalar_residual': float(np.max(data['stages/scalar_residual'][:], initial=0.)),
                   'max_weak_continuity': float(np.max(diagnostics['continuity_residual'])),
                   'max_quadrature_physical_divergence': float(np.max(diagnostics['divergence_inf'])),
                   'final_diagnostics': {key: float(values[-1]) for key, values in diagnostics.items()},
                   'snapshot_requested_times': data['snapshots/requested_times'][:].tolist(),
                   'snapshot_actual_times': data['snapshots/actual_times'][:].tolist(),
                   'display': display, 'animation': movie,
                   'developed_cycles': cycles, 'reference_comparison': comparison,
                   'qualification_note': 'Reference/developed-cycle gate only; overall qualification also requires '
                                         'explicit mesh/geometry and temporal sensitivity evidence.',
                   'limits': 'Affine polygonal cylinder; P2 velocity/P1 physical-pressure coefficients. '
                             'Only weak continuity is constrained, not pointwise divergence. '
                             'Stage pressures are increments. Physical pressure/forces use a separate '
                             'instantaneous constrained rate solve; no open-boundary energy theorem is claimed.'}
    write_json(report/'analysis.json', summary)
    write_json(report/'plots.json', {'display': display, 'animation': movie})
    with (report/'history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time', 'step', *diagnostics, *OBSERVATION_FIELDS[1:]])
        for index, time in enumerate(times):
            values = ([observations[key][index] for key in OBSERVATION_FIELDS[1:]]
                      if index < len(observations['time']) else ['']*(len(OBSERVATION_FIELDS)-1))
            writer.writerow([time, 0. if index == 0 else steps[index-1],
                             *[v[index] for v in diagnostics.values()], *values])
    if attempts:
        with (report/'attempts.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(ATTEMPT_FIELDS)
            writer.writerows(zip(*(attempts[key] for key in ATTEMPT_FIELDS)))
    if before != {p.name: digest(p) for p in directory.iterdir() if p.is_file()}:
        raise RuntimeError('Cylinder analysis input changed during readonly reporting')
    (report/'analysis.log').write_text(f"{manifest['status']}: readonly saved run {directory.name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Explicit saved cylinder run directory')
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(analyze(args.run, root=args.root))


if __name__ == '__main__':
    main()

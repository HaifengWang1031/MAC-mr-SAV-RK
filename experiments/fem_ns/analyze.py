"""Minimal diagnostics from an explicit saved FEM run; never launches computation."""
import argparse
import csv
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np

from experiments.records import PROJECT, digest, provenance, write_json
from .reconstruct import reconstruct
from .records import load_fem
from .attempts import FIELDS as ATTEMPT_FIELDS, STRINGS


def analyze(directory: Path, *, root: Path = PROJECT) -> Path:
    directory = Path(directory).resolve()
    root = Path(root).resolve()
    locator = directory.relative_to(root) if directory.is_relative_to(root) else directory
    cfg, manifest = load_fem(directory)
    report = (Path(root)/'reports'/'fem_ns'/directory.name).resolve()
    if report.is_relative_to(directory):
        raise ValueError('FEM reports must be separate from input runs')
    x, y = np.meshgrid(np.linspace(0., cfg['lx'], 17), np.linspace(0., cfg['ly'], 13))
    points = np.stack([x.ravel(), y.ravel()])
    fields = reconstruct(directory, points)
    speed = np.linalg.norm(fields['velocity'], axis=0)
    source = provenance()
    source['packages']['scikit-fem'] = version('scikit-fem')
    with h5py.File(directory/'results.h5', 'r') as data:
        times, dt = data['times'][:], data['dt'][:]
        diagnostics = {key: data['diagnostics/'+key][:] for key in data['diagnostics']}
        attempts = ({key: data['attempts/'+key].asstr()[:] if key in STRINGS else data['attempts/'+key][:]
                     for key in ATTEMPT_FIELDS} if 'attempts' in data else {})
        summary = {'run': str(locator), 'run_id': directory.name, 'status': manifest['status'],
                   'error': manifest.get('error', ''), 'final_time': fields['time'],
                   'accepted_steps': len(dt), 'r_final': fields['r'], 'mode': cfg.get('mode', 'fixed'),
                   'rejected_steps': int(np.sum(~attempts['accepted'])) if attempts else 0,
                   'controller': cfg.get('controller'), 'scalar_atol': cfg.get('scalar_atol'),
                   'manifest_sha256': digest(directory/'manifest.json'),
                   'results_sha256': manifest['results_sha256'], 'analysis_source': source,
                   'field_layout': cfg['field_layout'], 'pressure_semantics': cfg['pressure_semantics'],
                   'final_diagnostics': {key: float(values[-1]) for key, values in diagnostics.items()},
                   'sampled_max_speed': float(np.max(speed)), 'sample_shape': [13, 17],
                   'max_stage_residual': float(np.max(data['stages/residual'][:], initial=0.)),
                   'max_scalar_residual': float(np.max(data['stages/scalar_residual'][:], initial=0.)),
                   'snapshot_requested_times': data['snapshots/requested_times'][:].tolist(),
                   'snapshot_actual_times': data['snapshots/actual_times'][:].tolist(),
                   'limits': 'Field extrema are display samples, physical divergence is quadrature sampled; '
                             'stage pressures are multipliers, not physical endpoint-pressure accuracy evidence.'}
    report.mkdir(parents=True, exist_ok=True)
    write_json(report/'analysis.json', summary)
    with (report/'history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time', 'dt', *diagnostics])
        for i, time in enumerate(times):
            writer.writerow([time, 0. if i == 0 else dt[i-1], *[values[i] for values in diagnostics.values()]])
    with (report/'fields.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        pressure = fields['stage_pressure']
        writer.writerow(['x', 'y', 'u', 'v', 'speed', *([] if pressure is None else
                         [f'stage_pressure_{i}' for i in range(len(pressure))])])
        for i in range(points.shape[1]):
            writer.writerow([*points[:, i], *fields['velocity'][:, i], speed[i],
                             *([] if pressure is None else pressure[:, i])])
    if attempts:
        with (report/'attempts.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(ATTEMPT_FIELDS)
            writer.writerows(zip(*(attempts[key] for key in ATTEMPT_FIELDS)))
    (report/'analysis.log').write_text(f"{manifest['status']}: readonly saved run {directory.name}\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Explicit saved FEM run directory')
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(analyze(args.run, root=args.root))


if __name__ == '__main__':
    main()

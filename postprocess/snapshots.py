"""Snapshot reports for explicit complete MAC records; never computes a flow."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from run_support.records import write_json
from run_support.source import PROJECT, digest, provenance
from .mac import read_run_fields
from .plotting import field_figure, history_figure


def report(run: Path, *, root: Path = PROJECT, name: str = 'forced_ns_speed',
           fields: tuple[str, ...] = ('speed',)) -> Path:
    """Verify saved data and write figures, snapshot statistics and analysis source.

    Descriptive experiment keys are copied when recorded, not invented or required.
    The input computation identity is never recalculated or rewritten.
    """
    cfg, grid, times, available, history = read_run_fields(run)
    unknown = [field for field in fields if field not in available]
    if unknown:
        raise ValueError(f'Unknown fields {unknown}')
    if not times:
        raise ValueError('No stored snapshots to report')
    directory = root/f'reports/{name}'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    (directory/'figures').mkdir(parents=True)
    (directory/'tables').mkdir()
    for field in fields:
        field_figure(grid, times, available[field], directory/f'figures/{field}.png', field)
    history_figure(history, directory/'figures/history.png')
    speed, vorticity = available['speed'], available['vorticity']
    rows = []
    for time in times:
        rows.append({'time': time, 'max_speed': float(speed[time].max()),
                     'rms_speed': float(np.sqrt(np.mean(speed[time]**2))),
                     'max_vorticity': float(np.abs(vorticity[time]).max()),
                     'mean_vorticity': float(vorticity[time].mean())})
    with (directory/'tables/speed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory/'analysis.json', {
        'status': 'complete', 'run': str(run.resolve()), 'run_id': run.name,
        'results_sha256': digest(run/'results.h5'),
        'config': {key: cfg[key] for key in ('experiment', 'nx', 'ny', 'nu', 'amplitude', 'force',
                                            'initial_condition', 'scheme', 'dt', 'T', 'snapshots') if key in cfg},
        'snapshots': rows,
        'figures': [f'figures/{field}.png' for field in fields]+['figures/history.png'],
        'source': provenance(task_paths=('postprocess',)),
    })
    return directory


def main() -> None:
    """CLI shared by python -m postprocess.snapshots and the historical tool path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--name', default='forced_ns_speed')
    parser.add_argument('--fields', nargs='+', default=['speed'], choices=['speed', 'vorticity'])
    arguments = parser.parse_args()
    print(report(arguments.run, root=arguments.root, name=arguments.name, fields=tuple(arguments.fields)))


if __name__ == '__main__':
    main()

"""Compare direct/FFT CPU cost on identical MAC channel problems and saved fields."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from solver.channel import ChannelGrid
from solver.schemes.registry import DEFAULT_SCHEMES
from experiments.records import PROJECT, digest, provenance, write_json
from .analyze import read_fields
from .problems import SCHEMES
from .run import load_channel, run

DEFAULTS: dict[str, Any] = {'grids': [[32, 24], [64, 48]], 'schemes': list(DEFAULT_SCHEMES),
                          'repeats': 3, 'T': .04, 'steps': 10}


def benchmark(config: dict | None = None, *, root: Path = PROJECT) -> Path:
    supplied = config or {}
    if set(supplied)-set(DEFAULTS): raise ValueError('Unknown channel benchmark configuration')
    cfg = {**DEFAULTS, **supplied}
    if not cfg['grids'] or not cfg['schemes'] or len(set(cfg['schemes'])) != len(cfg['schemes']) \
            or any(s not in SCHEMES for s in cfg['schemes']) \
            or not np.isfinite(cfg['T']) or cfg['T'] <= 0:
        raise ValueError('Invalid benchmark grid/schemes/time')
    for key in ('repeats', 'steps'):
        if type(cfg[key]) is not int or cfg[key] < 1: raise ValueError(f'Invalid {key}')
    for nx, ny in cfg['grids']: ChannelGrid(nx, ny)
    if len({tuple(g) for g in cfg['grids']}) != len(cfg['grids']): raise ValueError('Duplicate benchmark grid')
    directory = root/'runs'/'channel_ns_benchmark'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    write_json(directory/'config.json', cfg)
    manifest = {'experiment': 'channel_ns_benchmark', 'status': 'running', 'source': provenance(),
                'config_sha256': digest(directory/'config.json')}
    write_json(directory/'manifest.json', manifest)
    (directory/'run.log').write_text('Same-MAC direct/FFT benchmark started\n')
    pairs: list[dict[str, Any]] = []
    try:
        for repeat in range(cfg['repeats']):
            for nx, ny in cfg['grids']:
                for name in cfg['schemes']:
                    members = {}
                    for backend in (('direct', 'fft') if repeat % 2 == 0 else ('fft', 'direct')):
                        path = run({'case': 'perturbed', 'backend': backend, 'scheme': name,
                                    'nx': nx, 'ny': ny, 'T': cfg['T'], 'dt': cfg['T']/cfg['steps'],
                                    'snapshots': []}, root=directory, rerun=True)
                        load_channel(path, require_complete=True)
                        members[backend] = str(path.relative_to(directory))
                    pairs.append({'nx': nx, 'ny': ny, 'scheme': name, 'repeat': repeat, **members})
                print(f'channel benchmark repeat={repeat+1}/{cfg["repeats"]} grid={nx}x{ny}', flush=True)
                with (directory/'run.log').open('a') as log:
                    log.write(f'repeat={repeat+1}/{cfg["repeats"]} grid={nx}x{ny}\n')
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(directory/'results.json', {'pairs': pairs})
        manifest['results_sha256'] = digest(directory/'results.json')
        write_json(directory/'manifest.json', manifest)
        with (directory/'run.log').open('a') as log:
            log.write(f"{manifest['status']}: {manifest.get('error', '')}\n")
    return directory


def analyze_benchmark(directory: Path, *, root: Path = PROJECT) -> Path:
    directory = directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('experiment') != 'channel_ns_benchmark' or manifest.get('status') != 'complete':
        raise ValueError('Complete channel benchmark required')
    if digest(directory/'config.json') != manifest['config_sha256'] \
            or digest(directory/'results.json') != manifest['results_sha256']:
        raise ValueError('Channel benchmark checksum mismatch')
    cfg = json.loads((directory/'config.json').read_text())
    pairs = json.loads((directory/'results.json').read_text())['pairs']
    if len(pairs) != len(cfg['grids'])*len(cfg['schemes'])*cfg['repeats']:
        raise ValueError('Incomplete benchmark matrix')
    rows = []
    differences = []
    for pair in pairs:
        grid = ChannelGrid(pair['nx'], pair['ny'], 2*np.pi, 1.)
        fields = {}
        configs = {}
        for backend in ('direct', 'fft'):
            path = directory/pair[backend]
            child, saved = load_channel(path, require_complete=True)
            if saved['source'] != manifest['source'] \
                    or (child['backend'], child['scheme'], child['nx'], child['ny'], child['T'], child['case']) != \
                       (backend, pair['scheme'], grid.nx, grid.ny, cfg['T'], 'perturbed') \
                    or len(child['actual_steps']) != cfg['steps']:
                raise ValueError('Benchmark member/source mismatch')
            _, fields[backend] = read_fields(path)
            configs[backend] = {k: v for k, v in child.items() if k != 'backend'}
            rows.append({'nx': grid.nx, 'ny': grid.ny, 'scheme': pair['scheme'],
                         'repeat': pair['repeat'], 'backend': backend, **saved['metrics']})
        if configs['direct'] != configs['fft']: raise ValueError('Benchmark compares different effective cases')
        np.testing.assert_allclose(fields['direct']['times'], fields['fft']['times'], rtol=0., atol=1e-14)
        a = grid.pack(fields['direct']['u'], fields['direct']['v'])
        b = grid.pack(fields['fft']['u'], fields['fft']['v'])
        difference = grid.norm(a-b)
        if difference > 1e-9*(1+grid.norm(a)): raise ValueError('Backends disagree on benchmark field')
        differences.append(difference)
    aggregate = []
    for nx, ny in cfg['grids']:
        for name in cfg['schemes']:
            row: dict[str, Any] = {'nx': nx, 'ny': ny, 'scheme': name, 'repeats': cfg['repeats']}
            for backend in ('direct', 'fft'):
                group = [r for r in rows if (r['nx'], r['ny'], r['scheme'], r['backend']) == (nx, ny, name, backend)]
                if {r['repeat'] for r in group} != set(range(cfg['repeats'])): raise ValueError('Missing repeat')
                for metric in ('setup_cpu_seconds', 'integration_cpu_seconds', 'factorization_wall_seconds', 'linear_solve_wall_seconds'):
                    row[backend+'_'+metric] = float(np.median([r[metric] for r in group]))
                totals = [r['setup_cpu_seconds']+r['integration_cpu_seconds'] for r in group]
                row[backend+'_total_cpu_seconds'] = float(np.median(totals))
                row[backend+'_total_cpu_min'] = min(totals); row[backend+'_total_cpu_max'] = max(totals)
            row['integration_cpu_speedup'] = row['direct_integration_cpu_seconds']/row['fft_integration_cpu_seconds']
            row['total_cpu_speedup'] = row['direct_total_cpu_seconds']/row['fft_total_cpu_seconds']
            aggregate.append(row)
    report = root/'reports'/'channel_ns_benchmark'/directory.name
    report.mkdir(parents=True, exist_ok=True)
    write_json(report/'analysis.json', {'run': str(directory), 'manifest_sha256': digest(directory/'manifest.json'),
                                       'analysis_source': provenance(), 'timings': aggregate,
                                       'max_velocity_L2_backend_difference': max(differences),
                                       'thread_environment': manifest['source']['threads'],
                                       'limits': ['same MAC discretisation and short perturbed-channel run',
                                                  'warmup excluded; factorization already included in integration',
                                                  'not a Fourier-spectral/MAC accuracy comparison',
                                                  'no long-time turbulence performance claim']})
    with (report/'timings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--analyze', type=Path)
    args = parser.parse_args()
    print(analyze_benchmark(args.analyze, root=args.root) if args.analyze else
          benchmark(None if args.config is None else json.loads(args.config.read_text()), root=args.root))


if __name__ == '__main__': main()

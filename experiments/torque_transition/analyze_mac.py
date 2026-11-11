"""Read-only MAC preflight audit using saved faces and independent flux differences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.records import digest, write_json


def audit_mac(directory: Path) -> dict[str, Any]:
    """Verify committed artifacts, reconstruct all cell fluxes without a model/solver."""
    record = json.loads((directory/'result.json').read_text())
    if record['status'] != 'complete' or record.get('discretization') != 'mac':
        raise ValueError('Requires a complete MAC probe')
    for filename, checksum in record['artifacts'].items():
        if digest(directory/filename) != checksum:
            raise ValueError(f'Artifact checksum mismatch: {filename}')
    with np.load(directory/'final.npz') as saved:
        n, t = int(saved['nodes']), float(saved['t'])
        packed = saved['velocity']
        if str(saved['discretization']) != 'mac' or packed.shape != (2*n*(n-1),):
            raise ValueError('Invalid MAC face layout')
    if record['grid_cells'] != n or record['final_t'] != t:
        raise ValueError('Metadata/field mismatch')
    u, v = np.zeros((n, n+1)), np.zeros((n+1, n))
    u[:, 1:-1], v[1:-1, :] = packed[:n*(n-1)].reshape(n, n-1), packed[n*(n-1):].reshape(n-1, n)
    h = 2*np.pi/n
    div = (u[:, 1:]-u[:, :-1]+v[1:, :]-v[:-1, :])/h
    energy = float(np.sum(packed**2)/(2*n*n))
    xc = (np.arange(n)+.5)*h-np.pi
    uc, vc = (u[:, 1:]+u[:, :-1])/2, (v[1:, :]+v[:-1, :])/2
    angular = float(np.mean(xc[None, :]*vc-xc[:, None]*uc))
    for key, value in (('energy', energy), ('angular_momentum', angular)):
        if not np.isclose(value, record['final_observables'][key], rtol=1e-12, atol=1e-15):
            raise ValueError(f'Independent {key} mismatch')
    with np.load(directory/'history.npz') as history:
        times = history['times']
        samples = json.loads((directory/'scalar_samples.json').read_text())['samples']
        for sample in samples:
            nearest = times[np.argmin(abs(times-sample['request_t']))]
            if nearest != sample['actual_t']:
                raise ValueError('Sample is not the nearest accepted node')
    return {'grid_cells': n, 'final_t': t, 'energy': energy, 'angular_momentum': angular,
            'final_divergence_inf': float(np.max(abs(div))),
            'final_divergence_rms': float(np.sqrt(np.mean(div**2))),
            'max_divergence_inf': record['max_physical_divergence'],
            'relative_divergence': record['final_observables']['relative_divergence'],
            'dissipation': record['final_observables']['dissipation'],
            'enstrophy': record['final_observables']['enstrophy'],
            'accepted_steps': record['accepted_steps'], 'attempts': record['attempts'],
            'wall_seconds': record['integration_wall_seconds'], 'u': u, 'v': v}


def face_difference(coarse: dict[str, Any], fine: dict[str, Any]) -> float:
    """Flux-preserving 2:1 face restriction; relative discrete velocity L2 difference."""
    if fine['grid_cells'] != 2*coarse['grid_cells'] or fine['final_t'] != coarse['final_t']:
        raise ValueError('Requires matching times and 2:1 grids')
    u = (fine['u'][::2, ::2]+fine['u'][1::2, ::2])/2
    v = (fine['v'][::2, ::2]+fine['v'][::2, 1::2])/2
    denominator = float(np.sum(u*u)+np.sum(v*v))
    return float(np.sqrt((np.sum((u-coarse['u'])**2)+np.sum((v-coarse['v'])**2))/denominator))


def analyze(directory: Path, output: Path) -> dict[str, Any]:
    """Persist a bounded-study report; never infer long-run qualification or launch jobs."""
    output.mkdir(parents=True, exist_ok=False)
    names = ('mac128-x4-t1', 'mac256-x4-t1', 'mac512-x4-t1',
             'mac256-x4-t10', 'mac256-x0-t1', 'mac256-x12-t1')
    fields = {name: audit_mac(directory/'runs'/name) for name in names}
    inputs = {str(directory/'runs'/name/'result.json'): digest(directory/'runs'/name/'result.json')
              for name in names}
    performance = []
    for name in ('mac256-concurrency', 'mac256-concurrency-finalists'):
        path = directory/'runs'/name/'summary.json'
        inputs[str(path)] = digest(path)
        performance += [{'round': name, **row} for row in json.loads(path.read_text())['rows']]
    differences = {name: face_difference(fields[f'mac{coarse}-x4-t1'], fields[f'mac{fine}-x4-t1'])
                   for name, coarse, fine in (('128_to_256', 128, 256), ('256_to_512', 256, 512))}
    result = {'status': 'complete_short_mac_preflight', 'analysis_sha256': digest(Path(__file__)),
              'inputs': inputs, 'performance': performance, 'differences_t1': differences,
              'probes': {name: {k: v for k, v in field.items() if k not in ('u', 'v')}
                         for name, field in fields.items()},
              'production': {'discretization': 'mac', 'grid_cells': 256, 'T': 100000.,
                             'scalar_interval': 1., 'field_interval': 10., 'approved': False},
              'limits': ['Only X4 reaches t10; grid comparison is at t1, not a developed-flow qualification.',
                         'No stationary statistics, temporal convergence or 256-grid long-time qualification.',
                         'No production streaming/PI-history checkpointing; no formal launch.',
                         'Short common-seed concurrency is not a heterogeneous 13-case makespan benchmark.']}
    write_json(output/'analysis.json', result)
    lines = ['# EIT-WSL 256² MAC preflight', '', '**T=100000 production not approved or started.**', '',
             '## Native probes', '', '| Probe | E | L | max full divergence | Steps / attempts | Wall / s |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in fields.items():
        lines.append(f"| {name} | {row['energy']:.9f} | {row['angular_momentum']:.9f} | "
                     f"{row['max_divergence_inf']:.3e} | {row['accepted_steps']}/{row['attempts']} | "
                     f"{row['wall_seconds']:.3f} |")
    lines += ['', '## Face-restricted velocity differences at t=1', '']
    lines += [f'- {name}: {value:.4%}' for name, value in differences.items()]
    lines += ['', '## Three-repeat native concurrency', '',
              '| Round | Workers | Threads | Median throughput | Peak RSS sum / MiB |',
              '|---|---:|---:|---:|---:|---:|']
    for row in performance:
        lines.append(f"| {row['round']} | {row['workers']} | {row['threads']} | "
                     f"{row['median_throughput']:.6f} | {row['max_peak_rss_bytes']/2**20:.1f} |")
    lines += ['', '## Limits', '']+[f'- {item}' for item in result['limits']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.output)

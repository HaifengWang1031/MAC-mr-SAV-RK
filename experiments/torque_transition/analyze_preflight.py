"""Read-only audit of explicit preflight artifacts; does not import a solver model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import legendre as leg

from experiments.records import digest, write_json


def reconstruct(path: Path, nodes: int = 256) -> dict[str, Any]:
    """Independent numpy.polynomial reconstruction on a common audit Gauss grid."""
    with np.load(path) as data:
        velocity = data['velocity']
        t = float(data['t'])
    size = int(np.sqrt(len(velocity)/2))
    xi, weights = leg.leggauss(nodes)
    x = np.pi*(xi+1)
    coefficients = np.zeros((size+2, size))
    indices = np.arange(size)
    coefficients[indices, indices] = 1.
    coefficients[indices+2, indices] = -1.
    basis = leg.legval(xi, coefficients)
    derivative = leg.legval(xi, leg.legder(coefficients, axis=0))/np.pi
    fields = velocity.reshape(2, size, size)
    u, v = (basis.T @ c.T @ basis for c in fields)
    ux, vx = (basis.T @ c.T @ derivative for c in fields)
    uy, vy = (derivative.T @ c.T @ basis for c in fields)
    weight = weights[:, None]*weights[None, :]/4
    xx, yy = np.meshgrid(x, x)
    div2 = float(np.sum(weight*(ux+vy)**2))
    grad2 = float(np.sum(weight*(ux*ux+uy*uy+vx*vx+vy*vy)))
    return {'t': t, 'modes': size, 'u': u, 'v': v, 'weight': weight,
            'energy': float(np.sum(weight*(u*u+v*v))/2),
            'angular_momentum': float(np.sum(weight*((xx-np.pi)*v-(yy-np.pi)*u))),
            'divergence_rms': float(np.sqrt(div2)),
            'relative_divergence': float(np.sqrt(div2/grad2)) if grad2 else 0.,
            'enstrophy': float(np.sum(weight*(vx-uy)**2)/2),
            'dissipation': 2e-4*grad2}


def analyze(directory: Path, output: Path) -> dict[str, Any]:
    """Report measured throughput and finite-time spatial sensitivity, not qualification."""
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    performance: list[dict[str, Any]] = []
    for name in ('concurrency-t10', 'concurrency-extension-retry', 'concurrency-finalists'):
        path = directory/'runs'/name/'summary.json'
        inputs[str(path)] = digest(path)
        data = json.loads(path.read_text())
        performance.extend({'round': name, **row} for row in data['rows'])
    fields = []
    space_rows = []
    for name in ('seed-x4-t10', 'space-q160-t10', 'space-q192-t10'):
        path = directory/'runs'/name/'final.npz'
        inputs[str(path)] = digest(path)
        metadata_path = path.with_name('result.json')
        metadata = json.loads(metadata_path.read_text())
        if metadata['status'] != 'complete' or metadata['case'] != 4 or metadata['final_t'] != 10.:
            raise ValueError(f'Incomplete or incompatible spatial input: {path}')
        for filename, checksum in metadata.get('artifacts', {}).items():
            if digest(path.parent/filename) != checksum:
                raise ValueError(f'Artifact checksum mismatch: {path.parent/filename}')
        item = reconstruct(path)
        fields.append(item)
        space_rows.append({k: v for k, v in item.items() if k not in ('u', 'v', 'weight')})
    backends = []
    for backend in ('tensor', 'direct'):
        path = directory/'runs'/f'backend-{backend}'/'result.json'
        inputs[str(path)] = digest(path)
        data = json.loads(path.read_text())
        if data['status'] != 'complete':
            raise ValueError(f'Incomplete backend probe: {path}')
        backends.append({'backend': backend,
                         'wall_seconds': data['integration_wall_seconds'],
                         'accepted_steps': data['accepted_steps'],
                         'peak_rss_bytes': data['peak_rss_bytes']})
    reference = fields[-1]
    for row, item in zip(space_rows, fields):
        difference2 = float(np.sum(reference['weight']*((item['u']-reference['u'])**2+
                                                       (item['v']-reference['v'])**2)))
        row['relative_velocity_difference_to_192'] = float(np.sqrt(difference2/(2*reference['energy'])))
    result: dict[str, Any] = {'status': 'complete_with_unqualified_numerics', 'inputs': inputs,
              'analysis_source_sha256': digest(Path(__file__)),
              'performance': performance, 'backends': backends,
              'space_audit_gauss_nodes': 256, 'space': space_rows,
              'production': {'T': 100000., 'gauss_nodes': 128, 'scalar_interval': 1.,
                             'field_interval': 10., 'approved': False},
              'limits': ['Only X4 t=10..10.2 is used for concurrency; not a stationary flow benchmark.',
                         'Physical divergence and spatial sensitivity remain unresolved.',
                         'No time-tolerance study, strict PI resume or streaming production implementation.',
                         'Peak throughput is not the makespan of a finite heterogeneous 13-job batch.',
                         'Initial T100 probe interrupted; extension attempt failed on missing provenance file, fixed before retry.',
                         'Round1 predates additive provenance fields; source retained as preflight-round1.py.']}
    write_json(output/'analysis.json', result)
    lines = ['# EIT-WSL torque preflight', '', '**Production not approved or started.**', '',
             '## Concurrency (three interleaved repeats per row)', '',
             '| Round | Workers | Threads | Median physical-time / wall second | Peak RSS sum / MiB |',
             '|---|---:|---:|---:|---:|']
    for row in performance:
        lines.append(f"| {row['round']} | {row['workers']} | {row['threads']} | "
                     f"{row['median_throughput']:.6f} | {row['max_peak_rss_bytes']/2**20:.1f} |")
    lines += ['', '## Backend same-state single probe (t=10 to 10.02)', '',
              '| Backend | Wall seconds | Accepted steps | Peak RSS / MiB |',
              '|---|---:|---:|---:|']
    for row in backends:
        lines.append(f"| {row['backend']} | {row['wall_seconds']:.6f} | {row['accepted_steps']} | "
                     f"{row['peak_rss_bytes']/2**20:.1f} |")
    lines += ['', '## Common 256-point independent reconstruction at t=10', '',
              '| Modes | Energy | Angular momentum | RMS divergence | div/grad L2 | Velocity difference to q192 |',
              '|---:|---:|---:|---:|---:|---:|']
    for row in space_rows:
        lines.append(f"| {row['modes']} | {row['energy']:.8f} | {row['angular_momentum']:.8f} | "
                     f"{row['divergence_rms']:.6f} | {row['relative_divergence']:.4%} | "
                     f"{row['relative_velocity_difference_to_192']:.4%} |")
    lines += ['', '## Limits', '']+[f'- {item}' for item in result['limits']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.output)

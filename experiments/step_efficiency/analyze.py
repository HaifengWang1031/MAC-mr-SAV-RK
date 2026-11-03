"""Read explicit warm-step run IDs; never construct a model or launch computation."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.records import PROJECT, digest, provenance, write_json

LABELS = {'mac_direct': 'MAC sparse LU', 'mac_tensor': 'MAC tensor + Schur-CG',
          'legendre_direct': 'Legendre sparse LU', 'legendre_tensor': 'Legendre tensor + Schur-CG',
          'channel_direct': 'Channel sparse LU', 'channel_fft': 'Channel FFT + band LU',
          'fourier_leray': 'Fourier-Leray'}
PAIR_GROUPS = (('mac_direct', 'mac_tensor'), ('legendre_direct', 'legendre_tensor'),
               ('channel_direct', 'channel_fft'))


def analyze(directory: Path, *, root: Path = PROJECT) -> Path:
    directory = directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('experiment') != 'step_efficiency' or manifest.get('status') != 'complete':
        raise ValueError('Complete step efficiency run required')
    for filename, checksum_key in (('config.json', 'config_sha256'), ('results.json', 'results_sha256')):
        if digest(directory/filename) != manifest[checksum_key]:
            raise ValueError(f'Checksum mismatch: {filename}')
    cfg = json.loads((directory/'config.json').read_text())
    rows = json.loads((directory/'results.json').read_text())['rows']
    expected = {(backend, name, schedule, repeat) for backend in cfg['backends']
                for name in cfg['schemes'] for schedule in ('fixed', 'variable')
                for repeat in range(cfg['repeats'])}
    keys = [(r['backend'], r['scheme'], r['schedule'], r['repeat']) for r in rows]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('Incomplete/duplicate benchmark matrix')
    saved: dict[tuple[str, str, str, int], dict[str, np.ndarray]] = {}
    measurements: list[dict[str, Any]] = []
    for case_key, row in zip(keys, rows):
        if digest(directory/row['fields']) != row['fields_sha256']:
            raise ValueError(f'Checksum mismatch: {row["fields"]}')
        with np.load(directory/row['fields'], allow_pickle=False) as data:
            fields = {name: data[name] for name in data.files}
        if any(not np.isfinite(array).all() for array in fields.values()):
            raise ValueError('Nonfinite saved field/timing')
        steps, stages = cfg['steps'], row['n_stages']
        if fields['timing'].shape != (steps, 4) or np.any(fields['timing'] <= 0) \
                or fields['checks'].shape != (steps, stages, 4) \
                or fields['pressure'].shape[0] != stages:
            raise ValueError('Invalid measured step/stage shapes')
        expected_times = np.r_[0., np.cumsum(cfg['actual_steps'][row['schedule']])]
        np.testing.assert_array_equal(fields['times'], expected_times)
        if fields['times'][-1] != row['final_time'] or fields['r'].item() != row['final_r']:
            raise ValueError('Saved endpoint metadata mismatch')
        if fields['checks'][:, :, :2].max() > 100*cfg['stokes_tolerance']:
            raise ValueError('Full momentum/continuity check failed')
        factors = row['counters'].get('factorizations')
        if row['backend'] in ('mac_direct', 'legendre_direct', 'channel_direct', 'channel_fft') \
                and factors != (0 if row['schedule'] == 'fixed' else steps):
            raise ValueError('Warm/new-shift factorization accounting mismatch')
        timing = fields['timing']
        measurements.append({**row, 'step_cpu_ms': float(1000*timing[:, 0].mean()),
            'diagnostics_cpu_ms': float(1000*timing[:, 2].mean()),
            'full_cpu_ms': float(1000*(timing[:, 0]+timing[:, 2]).mean()),
            'full_wall_ms': float(1000*(timing[:, 1]+timing[:, 3]).mean()),
            'max_stage_residual': float(fields['checks'][:, :, 0].max()),
            'max_continuity_residual': float(fields['checks'][:, :, 1].max()),
            'max_physical_divergence': float(fields['checks'][:, :, 2].max())})
        saved[case_key] = {name: fields[name] for name in ('u', 'v', 'pressure', 'stage_r')}
    pairs = []
    for left, right in PAIR_GROUPS:
        if left not in cfg['backends'] or right not in cfg['backends']:
            continue
        for name in cfg['schemes']:
            for schedule in ('fixed', 'variable'):
                for repeat in range(cfg['repeats']):
                    a, b = saved[left, name, schedule, repeat], saved[right, name, schedule, repeat]
                    absolute = {key: float(np.linalg.norm(a[key]-b[key])) for key in a}
                    relative = {key: absolute[key]/max(float(np.linalg.norm(a[key])), 1e-30) for key in a}
                    # Roundoff in near-zero pressure/auxiliary increments needs an absolute floor.
                    for key in a:
                        if absolute[key] > 1e-10+1e-8*float(np.linalg.norm(a[key])):
                            raise ValueError(f'Backend disagreement: {left}/{right}/{name}/{key}')
                    pairs.append({'backends': [left, right], 'scheme': name, 'schedule': schedule,
                                  'repeat': repeat, 'absolute_difference': absolute,
                                  'relative_difference': relative})
    aggregate: list[dict[str, Any]] = []
    for backend in cfg['backends']:
        for name in cfg['schemes']:
            for schedule in ('fixed', 'variable'):
                members = [r for r in measurements if
                    (r['backend'], r['scheme'], r['schedule']) == (backend, name, schedule)]
                item: dict[str, Any] = {'backend': backend, 'scheme': name, 'schedule': schedule}
                for metric in ('step_cpu_ms', 'diagnostics_cpu_ms', 'full_cpu_ms', 'full_wall_ms'):
                    values = [r[metric] for r in members]
                    item[metric] = float(np.median(values))
                    item[metric+'_min'], item[metric+'_max'] = min(values), max(values)
                item['factorizations_per_step'] = float(np.median([
                    r['counters'].get('factorizations', 0)/cfg['steps'] for r in members]))
                item['pressure_iterations_per_step'] = float(np.median([
                    r['counters'].get('total_iterations', 0)/cfg['steps'] for r in members]))
                for metric in ('max_stage_residual', 'max_continuity_residual', 'max_physical_divergence'):
                    item[metric] = max(r[metric] for r in members)
                aggregate.append(item)
    limits = [
        'CPU median of repeat means; full step includes one native diagnostic call.',
        'Imports, JIT/loading, model/force setup, initial projection and warmup excluded.',
        'Fixed schedule reuses its warmed factor; each new variable shift is charged.',
        'Variable schedule is prescribed: no controller, rejected steps or step-selection overhead.',
        'N is grid points/cells for MAC/Channel/Fourier, velocity modes for Legendre (degree N+1).',
        'Different boundary conditions and initial/force profiles; no equal-accuracy or equal-problem ranking.',
        'Fourier external grid also uses strict 2/3 filtering; fewer active modes than grid points.',
        'Short smooth MMS; iterative counts depend on parameters and right-hand sides.',
    ]
    report = root/'reports'/'step_efficiency'/directory.name
    report.mkdir(parents=True, exist_ok=True)
    write_json(report/'analysis.json', {'run': str(directory), 'manifest_sha256': digest(directory/'manifest.json'),
        'config': cfg, 'run_source': manifest['source'], 'analysis_source': provenance(),
        'timings': aggregate, 'pairs': pairs, 'measurements': measurements, 'limits': limits})
    with (report/'timings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    lines = [f'# N={cfg["N"]} warm per-step efficiency', '', f'Run: `{directory.name}`', '',
             f'Domain {cfg["lx"]} × {cfg["ly"]}; nu={cfg["nu"]}; gamma={cfg["gamma"]}; '
             f'nominal dt={cfg["fixed_step"]}; {cfg["steps"]} measured steps × '
             f'{cfg["repeats"]} repeats, after {cfg["warmup_steps"]} warmup steps.', '',
             'Primary metric: CPU ms per complete step (scheme + one model diagnostic).', '']
    for schedule in ('fixed', 'variable'):
        lines.extend([f'## {schedule}', '', '| Backend | '+' | '.join(cfg['schemes'])+' |',
                      '|---|'+'---:|'*len(cfg['schemes'])])
        for backend in cfg['backends']:
            values = [next(r['full_cpu_ms'] for r in aggregate if
                (r['backend'], r['scheme'], r['schedule']) == (backend, name, schedule)) for name in cfg['schemes']]
            lines.append('| '+LABELS[backend]+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
        lines.append('')
    lines.extend(['## Interpretation limits', '']+['- '+item for item in limits])
    lines.extend(['', 'Backend equivalence checks apply within each discretisation, using saved final '
                  'velocity, stage pressure and auxiliary increments. Full stage residual and '
                  'continuity checks remain enabled. See analysis.json for all repeats, differences '
                  'and setup/warmup/counter details.'])
    (report/'comparison.md').write_text('\n'.join(lines)+'\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(analyze(args.run, root=args.root))


if __name__ == '__main__':
    main()

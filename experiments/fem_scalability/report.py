"""Read saved benchmark records only; no model construction or time stepping."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import scipy.sparse as sp


def field_errors(sample: Path, reference: Path, mass: Any) -> dict[str, float]:
    with np.load(sample.with_suffix('.npz')) as actual, np.load(reference.with_suffix('.npz')) as expected:
        if any(not np.isfinite(data[key]).all() for data in (actual, expected) for key in data.files):
            raise ValueError('Nonfinite saved benchmark field')
        difference = actual['velocity']-expected['velocity']
        result = dict(velocity_l2=float(np.sqrt(max(0., difference @ (mass @ difference))))/
                      (1+float(np.sqrt(max(0., expected['velocity'] @ (mass @ expected['velocity']))))),
                      velocity_inf=float(np.max(abs(difference)))/(1+float(np.max(abs(expected['velocity'])))))
        for key in ('stage_pressure', 'physical_pressure', 'stage_r'):
            if actual[key].shape != expected[key].shape:
                raise ValueError(f'Saved field shape differs: {key}')
            result[key] = (float(np.max(abs(actual[key]-expected[key])))/(1+float(np.max(abs(expected[key]))))
                           if actual[key].size and actual[key].shape == expected[key].shape else 0.)
        return result


def analyze(run: Path, output: Path) -> dict[str, Any]:
    campaign = json.loads((run/'campaign.json').read_text())
    for directory, key in [('candidate_source', 'candidate_hashes'), ('candidate_v2_source', 'candidate_v2_hashes')]:
        for relative, expected in campaign.get(key, {}).items():
            if hashlib.sha256((run/directory/relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Candidate source was modified after dispatch')
    baseline = json.loads((run/'baseline.json').read_text()) if (run/'baseline.json').exists() else {}
    for relative, expected in baseline.get('files', {}).items():
        if hashlib.sha256((run/'baseline_source'/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Original baseline source was modified after dispatch')
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((run/'samples').rglob('*.json')):
        fixture, workload, scheme, variant, repeat = path.relative_to(run/'samples').parts
        record = json.loads(path.read_text())
        references = [path.parent.parent/variant/repeat for variant in
                      ('original_direct', 'original_recycled', 'metis_recycled', 'mass_cg')]
        reference = next((p for p in references if p.exists() and
                          json.loads(p.read_text()).get('status') == 'complete'), references[0])
        reasons = []
        if record.get('exit_code', 0) != 0:reasons.append('worker process failed')
        for filename, key in [('mesh.npz', 'fixture_sha256'), ('initial.npz', 'initial_sha256')]:
            if key in record and hashlib.sha256((run/'fixtures'/fixture/filename).read_bytes()).hexdigest() != record[key]:
                raise ValueError('Saved benchmark fixture changed after measurement')
        if 'field_sha256' in record and hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest() != record['field_sha256']:
            raise ValueError('Saved benchmark field changed after measurement')
        if record['status'] != 'complete':reasons.append(record.get('error', record['status']))
        metrics = record.get('metrics', {})
        if metrics.get('stokes_petsc_fallbacks', 0) or metrics.get('pressure_petsc_fallbacks', 0):
            reasons.append('global LU fallback; not AMG-only')
        if any(pool.get('num_threads', 1) != 1 for pool in record.get('threadpools', [])):
            reasons.append('native thread pool differs from one')
        record['reference_sample'] = str(reference.relative_to(run)) if reference.exists() else None
        errors = {}
        if workload == 'matrix':
            rows = record.get('matrix_solves', [])
            if len(rows) != 5:reasons.append('incomplete shift regimes')
            if any(row['velocity_error_inf'] > 1e-7 or row['pressure_error_inf'] > 1e-7 for row in rows):
                reasons.append('manufactured solution error')
        elif record['status'] == 'complete' and reference.exists():
            ref = json.loads(reference.read_text())
            if ref['status'] != 'complete':reasons.append('reference failed')
            else:
                errors = field_errors(path, reference, sp.load_npz(run/'fixtures'/fixture/'mass.npz'))
                errors['r'] = abs(record['final_r']-ref['final_r'])
                same_schedule = len(record['times']) == len(ref['times']) and np.allclose(
                    record['times'], ref['times'], atol=1e-12, rtol=1e-10)
                record['same_schedule'] = bool(same_schedule)
                if not same_schedule and workload != 'adaptive':reasons.append('replay schedule changed')
                if errors['velocity_inf'] > 1e-8 or errors['r'] > 1e-8 or errors['physical_pressure'] > 1e-7:
                    reasons.append('endpoint field agreement')
                if same_schedule and (errors['stage_pressure'] > 1e-7 or errors['stage_r'] > 1e-8):
                    reasons.append('stage pressure/scalar agreement')
        elif workload != 'matrix':reasons.append('missing complete saved reference')
        record.update(sample=str(path.relative_to(run)), errors=errors,
                      qualified=not reasons, disqualification=reasons)
        groups[(fixture, workload, scheme, variant)].append(record)
    summaries: list[dict[str, Any]] = []
    for (fixture, workload, scheme, variant), rows in groups.items():
        qualified = all(row['qualified'] for row in rows)
        summary: dict[str, Any] = dict(fixture=fixture, workload=workload, scheme=scheme, variant=variant,
            repeats=len(rows), qualified=qualified, samples=rows,
            disqualification=sorted({reason for row in rows for reason in row['disqualification']}))
        for key in ('online_seconds', 'initialization_seconds', 'end_to_end_seconds', 'peak_rss_bytes',
                    'pressure_recovery_seconds'):
            values = [row[key] for row in rows if key in row]
            if values:summary[key] = dict(median=median(values), minimum=min(values), maximum=max(values))
        summaries.append(summary)
    for summary in summaries:
        baseline = next((s for s in summaries if s['fixture']==summary['fixture'] and
            s['workload']==summary['workload'] and s['scheme']==summary['scheme'] and
            s['variant']=='original_recycled' and s['qualified']), None)
        if summary['qualified'] and baseline:
            summary['online_speedup_vs_recycled'] = baseline['online_seconds']['median']/summary['online_seconds']['median']
            summary['end_to_end_speedup_vs_recycled'] = baseline['end_to_end_seconds']['median']/summary['end_to_end_seconds']['median']
    result: dict[str, Any] = dict(run=str(run), analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limits=campaign['limits'], summaries=summaries,
                  agreement_thresholds=dict(velocity_inf=1e-8, r=1e-8, pressure=1e-7),
                  notes=['Relative field errors use 1+reference norm; r uses absolute error.',
                         'Adaptive endpoint pressure is compared independently of differing stage nodes.',
                         'Closed-domain endpoint physical pressure is not reconstructed; pressure agreement there uses stage multipliers.',
                         'Worker and numerical sources are frozen; controller file hashes were captured at record write and routing helpers evolved between phases.',
                         'Large-mesh CG and METIS timing batches are separate; their cross-batch ranking is exploratory.',
                         'Peak RSS includes imports, assembly, all cached factors, PC and saved arrays.',
                         'Online includes stage checks and native accepted-state diagnostics; initialization and physical pressure recovery are separate.',
                         'End to end includes process launch/imports, mesh load, initialization, integration, pressure recovery and file persistence.'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# FEM scalability comparison', '', f'Explicit run: `{run}`', '',
        '| Fixture | Workload | Scheme | Backend | Repeats | Qualified | Setup (s) | Online (s) | End to end (s) | Peak MiB | Online speedup vs recycled |',
        '|---|---|---|---|---:|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        def value(key: str, scale: float = 1.) -> str:
            return f"{s[key]['median']/scale:.4f}" if key in s else '—'
        speedup = f"{s['online_speedup_vs_recycled']:.3f}" if 'online_speedup_vs_recycled' in s else '—'
        lines.append(f"| {s['fixture']} | {s['workload']} | {s['scheme']} | {s['variant']} | {s['repeats']} | {s['qualified']} | {value('initialization_seconds')} | {value('online_seconds')} | {value('end_to_end_seconds')} | {value('peak_rss_bytes', 2**20)} | {speedup} |")
    lines += ['', 'Limits:', '']+['- '+note for note in result['limits']+result['notes']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()

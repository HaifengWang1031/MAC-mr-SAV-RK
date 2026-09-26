"""Read an explicit four-scheme batch and produce two manuscript-style tables."""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.forced_ns_convergence.analyze import number, read_run
from experiments.workflow import PROJECT, digest, provenance, write_json
from solver.mac.grid import MACGrid
from solver.mac.operators import MACOperators

PAIRS = (('sdirk2', 'sdirk2_mrsav'), ('sdirk3', 'sdirk3_mrsav'))
LABELS = {'sdirk2': 'SDIRK2', 'sdirk2_mrsav': 'SDIRK2-mr-ccSAV',
          'sdirk3': 'SDIRK3', 'sdirk3_mrsav': 'SDIRK3-mr-ccSAV'}


def table(rows: list[dict], times: list[float], pair: tuple[str, str], norm: str,
          tau_base: float, reference_step: float, nx: int, ny: int) -> str:
    lines = [r'\begin{table}', r'\centering',
             rf'\caption{{Velocity {norm} errors and observed rates on a ${nx}\times{ny}$ no-slip MAC grid. '
             rf'The common SDIRK3 reference uses $\tau_{{\rm ref}}={reference_step:.8g}$. '
             rf'Trial target steps are $\tau_k={tau_base:g}\,2^{{-k}}$; '
             r'interval-end steps are shortened to hit observation times.}',
             rf'\label{{tab:no-slip-{pair[0]}-{norm.lower()}}}',
             r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{l ' + ' '.join(['cc cc']*len(times)) + '}', r'\toprule',
             '& ' + ' & '.join(rf'\multicolumn{{4}}{{c}}{{$T={time:g}$}}' for time in times) + r'\\',
             ' '.join(rf'\cmidrule(lr){{{2+4*j}-{5+4*j}}}' for j in range(len(times))),
             '& ' + ' & '.join(' & '.join(rf'\multicolumn{{2}}{{c}}{{{LABELS[s]}}}' for s in pair)
                                for _ in times) + r'\\',
             ' '.join(rf'\cmidrule(lr){{{2+2*j}-{3+2*j}}}' for j in range(2*len(times))),
             '$k$ & ' + ' & '.join(['Error & Rate & Error & Rate']*len(times)) + r'\\',
             r'\midrule']
    for level in sorted({row['k'] for row in rows}):
        cells = [f'{level:g}']
        for time in times:
            for scheme in pair:
                row = next(row for row in rows if row['k'] == level and
                           row['time'] == time and row['scheme'] == scheme)
                rate = row[norm+'_rate']
                cells.extend([number(row[norm]), '--' if not np.isfinite(rate) else f'{rate:.2f}'])
        lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}}', r'\end{table}'])+'\n'


def analyze_batch(batch_path: Path, *, root: Path = PROJECT) -> Path:
    batch = json.loads(batch_path.read_text())
    config = batch['config']
    levels = config['k_levels']
    expected = {(level, scheme) for level in levels for pair in PAIRS for scheme in pair}
    if {(m['k'], m['scheme']) for m in batch['members']} != expected or \
            len(batch['members']) != len(expected):
        raise ValueError('Incomplete or duplicate comparison members')
    references = [read_run(Path(batch['references'][key])) for key in ('reference', 'refined')]
    members = [(item, read_run(Path(item['path']))) for item in batch['members']]
    common = references[0]['config']
    keys = ('experiment', 'nx', 'ny', 'lx', 'ly', 'nu', 'amplitude', 'gamma',
            'T', 'snapshots', 'force', 'initial_condition', 'boundary')
    for entry in references[1:]+[record for _, record in members]:
        if any(entry['config'][key] != common[key] for key in keys):
            raise ValueError('Mismatched physical or grid configuration')
        if entry['manifest']['source']['code_sha256'] != references[0]['manifest']['source']['code_sha256']:
            raise ValueError('Mixed solver source versions')
    if any(entry['manifest']['status'] != 'complete' or entry['config']['scheme'] != 'sdirk3'
           for entry in references):
        raise ValueError('Both SDIRK3 references must complete')
    grid = MACGrid(common['nx'], common['ny'], common['lx'], common['ly'])
    operator = MACOperators(grid)

    def norms(difference: np.ndarray) -> dict[str, float]:
        return {'L2': grid.norm(difference),
                'H1': float(np.sqrt(max(grid.inner(difference, operator.K@difference), 0.)))}

    times = common['snapshots']
    gaps = {time: norms(references[0]['fields'][time]-references[1]['fields'][time])
            for time in times}
    rows: list[dict] = []
    for item, entry in members:
        if entry['config']['scheme'] != item['scheme']:
            raise ValueError('Scheme membership mismatch')
        for time in times:
            values = (norms(entry['fields'][time]-references[0]['fields'][time])
                      if time in entry['fields'] else {'L2': float('nan'), 'H1': float('nan')})
            rows.append({'k': item['k'], 'tau_target': config['tau_base']*2.**(-item['k']),
                         'scheme': item['scheme'], 'time': time, **values,
                         'status': entry['manifest']['status'],
                         'failure': entry['manifest'].get('error', ''),
                         'max_abs_r': entry['max_r'], 'run_id': Path(item['path']).name})
    for row in rows:
        earlier = next((other for other in rows if other['k'] < row['k'] and
                        other['scheme'] == row['scheme'] and other['time'] == row['time'] and
                        not any(row['k'] > level > other['k'] for level in levels)), None)
        for norm in ('L2', 'H1'):
            row[norm+'_rate'] = (float(np.log(earlier[norm]/row[norm]) /
                                      np.log(earlier['tau_target']/row['tau_target']))
                                  if earlier is not None and
                                  np.isfinite([earlier[norm], row[norm]]).all() and
                                  min(earlier[norm], row[norm]) > 0 else float('nan'))
    checks = []
    for time in times:
        for norm in ('L2', 'H1'):
            finite = [row[norm] for row in rows if row['time'] == time and
                      np.isfinite(row[norm]) and row[norm] > 0]
            ratio = gaps[time][norm]/min(finite) if finite else float('inf')
            checks.append({'time': time, 'norm': norm, 'reference_gap': gaps[time][norm],
                           'gap_over_min_error': ratio, 'passed': ratio <= config['sensitivity_threshold']})
    folder = root/'reports/no_slip_reference_convergence'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    (folder/'tables').mkdir(parents=True)
    write_json(folder/'input_batch.json', batch)
    analysis = {'status': 'running', 'batch': str(batch_path.resolve()),
                'batch_sha256': digest(batch_path), 'source': provenance(),
                'reference_checks': checks, 'reference_passed': all(c['passed'] for c in checks)}
    write_json(folder/'analysis.json', analysis)
    try:
        with (folder/'tables/errors.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for pair in PAIRS:
            for norm in ('L2', 'H1'):
                (folder/f'tables/{pair[0]}_{norm}.tex').write_text(
                    table(rows, times, pair, norm, config['tau_base'],
                          config['tau_base']*2.**(-config['reference_k']),
                          common['nx'], common['ny']))
        document = (r'\documentclass{article}'+'\n'+r'\usepackage{booktabs,graphicx}'+'\n'+
                    r'\begin{document}'+'\n'+
                    '\n'.join(rf'\input{{tables/{pair[0]}_{norm}.tex}}' for pair in PAIRS
                              for norm in ('L2', 'H1'))+'\n'+r'\end{document}'+'\n')
        (folder/'convergence.tex').write_text(document)
        import subprocess
        with (folder/'analysis.log').open('w') as log:
            completed = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                                        'convergence.tex'], cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                       check=False)
        if completed.returncode:
            raise RuntimeError('LaTeX compilation failed; see analysis.log')
        analysis['status'] = 'complete'
    except Exception as exc:
        analysis.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(folder/'analysis.json', analysis)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    print(analyze_batch(args.batch))

"""Execute the public result notebook against relocatable, known saved records."""
import csv
import hashlib
import json
from pathlib import Path

import h5py
import pytest

nbformat = pytest.importorskip('nbformat', reason='Install the optional notebook dependency group')
nbclient = pytest.importorskip('nbclient', reason='Install the optional notebook dependency group')

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT/'experiments/manufactured_convergence/studies/time_convergence/results.ipynb'


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved_study(root):
    report = root/'reports/study'
    (report/'tables').mkdir(parents=True)
    base = {'experiment': 'manufactured_ns', 'nx': 8, 'ny': 8, 'lx': 1., 'ly': 1.,
            'nu': .01, 'amplitude': .05, 'gamma': 1., 'T': 2.}
    batch = {'status': 'running', 'config': {'base': base, 'tau_base': .1,
                                          'k_levels': [0, 1]},
             'source': {'code_sha256': 'known-solver-source'}, 'members': []}
    rows, inputs = [], []
    for scheme in ('sdirk2', 'sdirk2_mrsav', 'sdirk3', 'sdirk3_mrsav'):
        for k, dt in ((0, .1), (1, .05)):
            run_id = f'{scheme}-{k}'
            run = root/'runs/manufactured_ns'/run_id
            run.mkdir(parents=True)
            config = {**base, 'scheme': scheme, 'dt': dt}
            (run/'config.json').write_text(json.dumps(config))
            with h5py.File(run/'results.h5', 'w') as data:
                data.attrs['status'] = 'complete'
                data.create_dataset('final/t', data=2.)
            manifest = {'status': 'complete', 'source': batch['source'],
                        'config_sha256': _digest(run/'config.json'),
                        'results_sha256': _digest(run/'results.h5')}
            (run/'manifest.json').write_text(json.dumps(manifest))
            saved = f'/old/machine/runs/manufactured_ns/{run_id}'
            batch['members'].append({'k': k, 'scheme': scheme, 'path': saved,
                                     'status': 'complete'})
            inputs.append({'path': saved, 'sha256': manifest['results_sha256']})
            second_order = scheme.startswith('sdirk2')
            error = ([.04, .01] if second_order else [.008, .001])[k]
            rows.append({'k': k, 'dt': dt, 'scheme': scheme, 'status': 'complete',
                         'L2': error, 'H1': error*10,
                         'L2_rate': 'nan' if k == 0 else 2 if second_order else 3,
                         'H1_rate': 'nan' if k == 0 else 2 if second_order else 3,
                         'run': saved, 'results_sha256': manifest['results_sha256']})
    (report/'input_batch.json').write_text(json.dumps(batch))
    analysis = {'status': 'complete', 'batch': '/old/machine/batches/known-batch/batch.json',
                'batch_sha256': _digest(report/'input_batch.json'),
                'source': batch['source'], 'inputs': inputs,
                'error_definition': 'MAC-consistent L2 and K-based H1 seminorm'}
    (report/'analysis.json').write_text(json.dumps(analysis))
    # Deliberately unsorted: the user-facing table and plots must order by k.
    with (report/'tables/errors.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(reversed(rows))
    return report


def _execute(root, report):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    parameters = next(cell for cell in notebook.cells
                      if 'parameters' in cell.metadata.get('tags', []))
    parameters.source = (f'from pathlib import Path\nPROJECT_ROOT = Path({str(root)!r})\n'
                         f'REPORT_DIRECTORY = Path({str(report.relative_to(root))!r})\n'
                         f'EXPECTED_ANALYSIS_SHA256 = {_digest(report/"analysis.json")!r}\n'
                         f'EXPECTED_TABLE_SHA256 = {_digest(report/"tables/errors.csv")!r}\n')
    guard = f'''import sys
sys.path.insert(0, {str(PROJECT)!r})
import experiments.workflow as workflow
import solver.integrate as integration
import solver.adaptivity as adaptivity
def forbidden(*args, **kwargs):
    raise AssertionError("Result presentation must not start computation")
workflow.run_experiment = forbidden
integration.integrate = forbidden
adaptivity.integrate_adaptive = forbidden
'''
    notebook.cells.insert(0, nbformat.v4.new_code_cell(guard))
    return nbclient.NotebookClient(notebook, timeout=90, kernel_name='python3',
                                   resources={'metadata': {'path': str(root)}}).execute()


def test_notebook_redraws_known_saved_errors_without_computation_or_input_changes(tmp_path):
    report = _saved_study(tmp_path)
    protected = {p: _digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    notebook = _execute(tmp_path, report)
    nbformat.validate(notebook)
    outputs = [out for cell in notebook.cells if cell.cell_type == 'code'
               for out in cell.get('outputs', [])]
    assert not any(out.output_type == 'error' for out in outputs)
    images = [out.data['image/png'] for out in outputs if 'image/png' in out.get('data', {})]
    assert images and all(len(image) > 1000 for image in images)
    html = '\n'.join(out.get('data', {}).get('text/html', '') for out in outputs)
    assert 'sdirk2' in html and 'sdirk3_mrsav' in html
    assert '4.000e-02' in html and '1.000e-03' in html
    assert html.index('1.000e-01') < html.index('5.000e-02')
    text = '\n'.join(out.get('text', '') for out in outputs)
    assert 'known-solver-source' in text and 'known-batch' in text
    assert all(_digest(path) == checksum for path, checksum in protected.items())


@pytest.mark.parametrize('damage', ['missing', 'checksum', 'status', 'endpoint'])
def test_notebook_rejects_missing_corrupt_or_incompatible_saved_records(tmp_path, damage):
    report = _saved_study(tmp_path)
    run = tmp_path/'runs/manufactured_ns/sdirk2-0'
    if damage == 'missing':
        (run/'results.h5').unlink()
    elif damage == 'checksum':
        with (run/'results.h5').open('ab') as stream:
            stream.write(b'corrupt')
    else:
        with h5py.File(run/'results.h5', 'a') as data:
            if damage == 'status':
                data.attrs['status'] = 'failed'
            else:
                data['final/t'][()] = 1.
        # Consistent byte checksums do not make incompatible record content valid.
        checksum = _digest(run/'results.h5')
        manifest = json.loads((run/'manifest.json').read_text())
        manifest['results_sha256'] = checksum
        (run/'manifest.json').write_text(json.dumps(manifest))
        analysis = json.loads((report/'analysis.json').read_text())
        analysis['inputs'][0]['sha256'] = checksum
        (report/'analysis.json').write_text(json.dumps(analysis))
        with (report/'tables/errors.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if Path(row['run']).name == run.name:
                row['results_sha256'] = checksum
        with (report/'tables/errors.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(nbclient.exceptions.CellExecutionError):
        _execute(tmp_path, report)

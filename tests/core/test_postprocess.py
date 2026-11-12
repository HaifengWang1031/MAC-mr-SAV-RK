"""Public readonly seams against known saved arrays, not a research computation."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from run_support.records import begin_run, finish_run, save_result
from solver.core import State
from solver.integrate import Result


def stored_mac(root: Path, *, status: str = 'complete') -> Path:
    """A synthetic record with known face samples; not evidence of a PDE solution."""
    config = {'nx': 4, 'ny': 3, 'lx': 2., 'ly': 1.5, 'nu': .1,
              'scheme': 'imex_euler', 'T': .25 if status == 'complete' else .5, 'dt': .25,
              'actual_steps': [.25] if status == 'complete' else [.25, .25], 'snapshots': [0., .25]}
    source = {'code_sha256': 'historical-computation', 'packages': {'numpy': 'recorded'}}
    directory, manifest = begin_run(root/'runs', config, source, metadata={})
    u = np.array([[0., .5, .5, .5, 0.], [0., 1.5, 1.5, 1.5, 0.],
                  [0., 2.5, 2.5, 2.5, 0.]])
    v = np.array([[0., 0., 0., 0.], [.75, 2.25, 3.75, 5.25],
                  [.75, 2.25, 3.75, 5.25], [0., 0., 0., 0.]])
    initial, final = State(0., u, v), State(.25, 2*u, 2*v, .01)
    history = [{'kinetic': 1., 'modified_energy': 1., 'divergence_inf': .1, 'r': 0.},
               {'kinetic': 4., 'modified_energy': 4.1, 'divergence_inf': .2, 'r': .01}]
    result = Result(final, [0., .25], history, snapshot_requests=[0., .25],
                    snapshot_times=[0., .25], snapshots=[initial, final], status=status)
    save_result(directory/'results.h5', result, {}, n_stages=1)
    finish_run(directory, manifest, status=status, error='' if status == 'complete' else 'saved failure',
               metrics={})
    return directory


def test_reader_recovers_known_saved_mac_fields_without_experiment_metadata(tmp_path: Path) -> None:
    from postprocess.mac import read_run_fields

    directory = stored_mac(tmp_path)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    cfg, grid, times, fields, history = read_run_fields(directory)
    assert cfg['scheme'] == 'imex_euler' and (grid.nx, grid.ny) == (4, 3)
    assert times == [0., .25]
    centre_u = np.array([[.25, .5, .5, .25], [.75, 1.5, 1.5, .75], [1.25, 2.5, 2.5, 1.25]])
    centre_v = np.array([[.375, 1.125, 1.875, 2.625], [.75, 2.25, 3.75, 5.25],
                         [.375, 1.125, 1.875, 2.625]])
    expected_speed = np.hypot(centre_u, centre_v)
    np.testing.assert_array_equal(fields['speed'][0.], expected_speed)
    np.testing.assert_array_equal(fields['speed'][.25], 2*expected_speed)
    np.testing.assert_array_equal(fields['vorticity'][0.], np.ones((2, 3)))
    np.testing.assert_array_equal(fields['vorticity'][.25], 2*np.ones((2, 3)))
    np.testing.assert_array_equal(history['time'], [0., .25])
    np.testing.assert_array_equal(history['r'], [0., .01])
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_snapshot_report_records_known_statistics_and_analysis_source_only(tmp_path: Path, monkeypatch) -> None:
    import csv
    import json
    import matplotlib.pyplot as plt
    from postprocess import plotting
    from postprocess.snapshots import report

    monkeypatch.setenv('PATH', '')
    monkeypatch.setattr(plotting, 'LATEX_SEARCH_PATHS', ())
    directory = stored_mac(tmp_path)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with plt.rc_context({'text.usetex': True}):
        output = report(directory, root=tmp_path, name='known', fields=('speed', 'vorticity'))
        assert plt.rcParams['text.usetex'] is True
    assert output.is_relative_to(tmp_path/'reports/known')
    analysis = json.loads((output/'analysis.json').read_text())
    assert analysis['status'] == 'complete' and analysis['run_id'] == directory.name
    assert analysis['source']['source_schema'] == 2
    inventory = analysis['source']['source_files']
    assert inventory['postprocess/mac.py'] and inventory['postprocess/snapshots.py']
    assert not any(p.startswith(('experiments/', 'tools/')) for p in inventory)
    assert analysis['config']['nx'] == 4 and 'force' not in analysis['config']
    assert [r['time'] for r in analysis['snapshots']] == [0., .25]
    assert analysis['snapshots'][0]['max_vorticity'] == 1.
    assert analysis['snapshots'][1]['mean_vorticity'] == 2.
    with (output/'tables/speed.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and float(rows[1]['max_speed']) == 2*float(rows[0]['max_speed'])
    for name in analysis['figures']:
        assert (output/name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_diagnostics_reads_explicit_failed_prefix_without_reclassifying_it(tmp_path: Path, monkeypatch) -> None:
    import csv
    import json
    import pytest
    import matplotlib.pyplot as plt
    from postprocess import plotting
    from postprocess.diagnostics import analyze_runs
    from postprocess.snapshots import report

    with pytest.raises(ValueError, match='explicit run directories'):
        analyze_runs([], root=tmp_path)
    assert not (tmp_path/'reports').exists()
    directory = stored_mac(tmp_path, status='failed')
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match='Complete run required'):
        report(directory, root=tmp_path)
    monkeypatch.setenv('PATH', '')
    monkeypatch.setattr(plotting, 'LATEX_SEARCH_PATHS', ())
    with plt.rc_context({'text.usetex': True}):
        output = analyze_runs([directory], root=tmp_path)
        assert plt.rcParams['text.usetex'] is True
    analysis = json.loads((output/'analysis.json').read_text())
    assert analysis['status'] == 'complete'  # Analysis completed, the input run did not.
    assert analysis['inputs'][0]['run_id'] == directory.name
    assert analysis['source']['source_files']['postprocess/diagnostics.py']
    with (output/'tables/summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]['status'] == 'failed'
    assert float(rows[0]['final_time']) == .25 and float(rows[0]['max_abs_r']) == .01
    assert (output/'figures/diagnostics.png').is_file()
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert json.loads((directory/'manifest.json').read_text())['source']['code_sha256'] == 'historical-computation'


@pytest.mark.parametrize('damage,message', [
    ('checksum', 'checksum'), ('not_finalized', 'finalized'),
    ('snapshot_node', 'requested time'), ('layout', 'MAC'), ('shape', 'MAC'),
    ('count', 'Snapshot'), ('nonfinite_node', 'Snapshot'),
])
def test_reader_rejects_invalid_records_before_plotting(tmp_path: Path, damage: str, message: str) -> None:
    import json
    from postprocess.mac import read_run_fields
    from run_support.records import write_json
    from run_support.source import digest

    directory = stored_mac(tmp_path)
    manifest = json.loads((directory/'manifest.json').read_text())
    if damage == 'checksum':
        (directory/'results.h5').write_bytes(b'corrupted')
    else:
        if damage == 'not_finalized':
            manifest['status'] = 'running'
        elif damage == 'layout':
            cfg = json.loads((directory/'config.json').read_text())
            cfg['discretization'] = 'channel_mac'
            write_json(directory/'config.json', cfg)
            manifest['config_sha256'] = digest(directory/'config.json')
        else:
            with h5py.File(directory/'results.h5', 'r+') as data:
                if damage == 'shape':
                    del data['snapshots/u']
                    data['snapshots/u'] = np.zeros((2, 3, 4))
                elif damage == 'count':
                    del data['snapshots/actual_times']
                    data['snapshots/actual_times'] = [0.]
                else:
                    data['snapshots/actual_times'][1] = np.nan if damage == 'nonfinite_node' else .2
        manifest['results_sha256'] = digest(directory/'results.h5')
        write_json(directory/'manifest.json', manifest)
    with pytest.raises(ValueError, match=message):
        read_run_fields(directory)
    assert not (tmp_path/'reports').exists()


def test_snapshot_cli_runs_without_research_or_numerical_computation_modules(tmp_path: Path) -> None:
    import json
    import os
    import shutil
    import subprocess
    import sys
    from run_support.source import ENVIRONMENT_FILES, PROJECT

    directory = stored_mac(tmp_path/'input')
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    portable = tmp_path/'portable'
    for name in ('solver', 'run_support', 'postprocess'):
        shutil.copytree(PROJECT/name, portable/name, ignore=shutil.ignore_patterns('__pycache__', '*.nbc', '*.nbi'))
    for name in ENVIRONMENT_FILES:
        shutil.copy2(PROJECT/name, portable/name)
    script = '''
import importlib.abc, pathlib, runpy, sys
root, run, output = sys.argv[1:]
sys.path.insert(0, root)
assert not (pathlib.Path(root)/'experiments').exists()
assert not (pathlib.Path(root)/'tools').exists()
class ReadonlyOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'experiments', 'tools', 'tests'}:
            raise AssertionError('Research dependency: '+fullname)
        if fullname.startswith('solver.') and fullname not in {'solver.mac', 'solver.mac.grid'}:
            raise AssertionError('Numerical computation dependency: '+fullname)
sys.meta_path.insert(0, ReadonlyOnly())
sys.argv = ['postprocess.snapshots', '--run', run, '--root', output,
            '--name', 'portable', '--fields', 'speed', 'vorticity']
runpy.run_module('postprocess.snapshots', run_name='__main__')
'''
    completed = subprocess.run([sys.executable, '-I', '-c', script, str(portable),
                                str(directory), str(tmp_path/'output')], cwd=portable,
                               env={**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'},
                               capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stdout+'\n'+completed.stderr
    assert completed.stdout.strip(), 'The CLI did not publish its report path'
    output = Path(completed.stdout.strip())
    assert output.is_relative_to(tmp_path/'output/reports/portable')
    analysis = json.loads((output/'analysis.json').read_text())
    assert analysis['status'] == 'complete' and analysis['snapshots'][0]['mean_vorticity'] == 1.
    assert (output/'figures/speed.png').is_file() and (output/'figures/vorticity.png').is_file()
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before

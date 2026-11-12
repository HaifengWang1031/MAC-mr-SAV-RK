"""Public source-scope and record workflows, independent of experiment modules."""
from pathlib import Path

import pytest


def source_tree(root: Path) -> None:
    for name in ('pyproject.toml', 'uv.lock', '.python-version',
                 'solver/__init__.py', 'run_support/__init__.py',
                 'experiments/task.py', 'experiments/unrelated.py'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# original\n')


def test_explicit_source_tracks_core_and_task_but_not_unrelated_research(tmp_path):
    from run_support.source import provenance

    source_tree(tmp_path)
    source = provenance(root=tmp_path, task_paths=['experiments/task.py'])
    assert source['source_schema'] == 2
    assert source['source_files']['solver/__init__.py']
    assert source['source_files']['experiments/task.py']
    assert 'experiments/unrelated.py' not in source['source_files']
    (tmp_path/'experiments/unrelated.py').write_text('# unrelated change\n')
    assert provenance(root=tmp_path, task_paths=['experiments/task.py']) == source
    (tmp_path/'experiments/task.py').write_text('# changed task\n')
    assert provenance(root=tmp_path, task_paths=['experiments/task.py'])['code_sha256'] != source['code_sha256']
    core = provenance(root=tmp_path)
    (tmp_path/'postprocess').mkdir()
    (tmp_path/'postprocess/mac.py').write_text('# readonly analysis, not computation\n')
    assert provenance(root=tmp_path) == core
    assert provenance(root=tmp_path, task_paths=['postprocess']) != core
    (tmp_path/'run_support/__init__.py').write_text('# changed support\n')
    assert provenance(root=tmp_path)['code_sha256'] != core['code_sha256']


def test_public_record_workflow_preserves_saved_source(tmp_path):
    import numpy as np
    from run_support.records import begin_run, finish_run, load_record, reusable_run, save_result
    from solver.core import State
    from solver.integrate import Result

    # A historical source is used as recorded, not regenerated with today's policy.
    source = {'code_sha256': 'historical-source', 'packages': {'numpy': 'old-version'}}
    config = {'scheme': 'imex_euler'}
    directory, manifest = begin_run(tmp_path/'runs', config, source, metadata={'experiment': 'smoke'})
    result = Result(State(0., np.zeros((2, 3)), np.zeros((3, 2))), [0.], [{'kinetic': 0., 'r': 0.}])
    save_result(directory/'results.h5', result, {}, n_stages=1)
    finish_run(directory, manifest, status='complete', error='', metrics={})
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert load_record(directory, require_complete=True) == (config, manifest)
    assert manifest['source'] == source
    assert reusable_run(tmp_path/'runs', config, source, reader=load_record) == directory
    assert reusable_run(tmp_path/'runs', config, {'code_sha256': 'new'}, reader=load_record) is None
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    (directory/'config.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        load_record(directory)
    assert reusable_run(tmp_path/'runs', config, source, reader=load_record) is None


@pytest.mark.parametrize('missing', ['solver', 'run_support', 'uv.lock', 'experiments/task.py'])
def test_missing_source_inventory_fails_closed(tmp_path, missing):
    import shutil
    from run_support.source import provenance

    source_tree(tmp_path)
    path = tmp_path/missing
    shutil.rmtree(path) if path.is_dir() else path.unlink()
    with pytest.raises(ValueError, match='Missing source'):
        provenance(root=tmp_path, task_paths=['experiments/task.py'])


def test_inventory_is_portable_deduplicated_and_confined(tmp_path):
    import shutil
    from run_support.source import provenance

    first, second = tmp_path/'a', tmp_path/'b'
    source_tree(first)
    shutil.copytree(first, second)
    one = provenance(root=first, task_paths=['experiments/task.py'])
    assert provenance(root=second, task_paths=['experiments/task.py', 'solver']) == one
    assert provenance(root=first, task_paths=['solver', 'experiments/task.py', 'solver']) == one
    with pytest.raises(ValueError, match='escapes root'):
        provenance(root=first, task_paths=['../b/experiments/task.py'])
    (first/'escape.py').symlink_to(second/'experiments/task.py')
    with pytest.raises(ValueError, match='escapes root'):
        provenance(root=first, task_paths=['escape.py'])
    (first/'internal.py').symlink_to(first/'experiments/task.py')
    with pytest.raises(ValueError, match='symlink'):
        provenance(root=first, task_paths=['internal.py'])
    (first/'empty').mkdir()
    with pytest.raises(ValueError, match='Empty source'):
        provenance(root=first, task_paths=['empty'])


@pytest.mark.parametrize('target_inside', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_directory_symlinks_are_rejected_instead_of_silently_omitted(tmp_path, target_inside, nested):
    from run_support.source import provenance

    root = tmp_path/'core'
    source_tree(root)
    dependency = root/'shared' if target_inside else tmp_path/'external'
    dependency.mkdir()
    (dependency/'actual.py').write_text('value = 1\n')
    link = root/'solver/linked' if nested else root/'selected'
    link.symlink_to(dependency, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink|escapes root'):
        provenance(root=root, task_paths=[] if nested else ['selected'])


def test_records_and_provenance_work_without_experiment_tree(tmp_path):
    import shutil
    import subprocess
    import sys
    from run_support.source import PROJECT, ENVIRONMENT_FILES

    root = tmp_path/'core'
    for name in ('solver', 'run_support'):
        shutil.copytree(PROJECT/name, root/name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ENVIRONMENT_FILES:
        shutil.copy2(PROJECT/name, root/name)
    script = '''
import importlib.abc, pathlib, sys
sys.path.insert(0, sys.argv[1])
class NoResearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'experiments', 'tools'}:
            raise AssertionError('Research dependency: '+fullname)
sys.meta_path.insert(0, NoResearch())
from run_support.records import begin_run, save_result, finish_run, load_record
from run_support.source import provenance
from solver.core import State
from solver.integrate import Result
import numpy as np
root = pathlib.Path(sys.argv[1])
source = provenance()
assert source['source_schema'] == 2
assert all(not p.startswith(('experiments/', 'tools/')) for p in source['source_files'])
run, manifest = begin_run(root/'runs', {'test': True}, source, metadata={})
result = Result(State(0., np.zeros((2, 3)), np.zeros((3, 2))), [0.], [{'r': 0.}])
save_result(run/'results.h5', result, {}, n_stages=1)
finish_run(run, manifest, status='complete', error='', metrics={})
assert load_record(run, require_complete=True)[1]['source'] == source
'''
    subprocess.run([sys.executable, '-I', '-c', script, str(root)], cwd=root, check=True,
                   capture_output=True, text=True)

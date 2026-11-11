"""Compatibility edges kept separate from experiment-free core validation."""
import hashlib


def test_legacy_source_encoding_and_new_shared_dependencies(tmp_path):
    from run_support.source import legacy_provenance

    for name in ('pyproject.toml', 'uv.lock', '.python-version', 'solver/a.py',
                 'experiments/a.py', 'tools/a.py'):
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'x')
    # Historical ordered stream, independent of the new inventory discovery.
    golden = b'pyproject.tomlxuv.lockx.python-versionxsolver/a.pyxexperiments/a.pyxtools/a.pyx'
    old = legacy_provenance(root=tmp_path)
    assert old['code_sha256'] == hashlib.sha256(golden).hexdigest()
    assert 'source_schema' not in old
    (tmp_path/'run_support').mkdir()
    (tmp_path/'run_support/records.py').write_text('# shared implementation')
    assert legacy_provenance(root=tmp_path)['code_sha256'] != old['code_sha256']


def test_old_record_imports_read_new_record_without_rewriting(tmp_path):
    from experiments.records import begin_run, digest, finish_run, load_record
    from run_support.records import load_record as core_load

    source = {'code_sha256': 'old-source'}
    run, manifest = begin_run(tmp_path, {'old': True}, source, metadata={})
    (run/'results.h5').write_bytes(b'historical payload')
    finish_run(run, manifest, status='complete', error='', metrics={})
    before = digest(run/'manifest.json')
    assert load_record(run) == core_load(run) == ({'old': True}, manifest)
    assert digest(run/'manifest.json') == before
    assert manifest['source'] == source


def test_source_selected_worker_captures_shared_record_implementation(tmp_path):
    import shutil
    import subprocess
    import sys
    from experiments.fem_scalability.run import freeze
    from run_support.source import PROJECT, ENVIRONMENT_FILES

    root = tmp_path/'input'
    shutil.copytree(PROJECT/'run_support', root/'run_support', ignore=shutil.ignore_patterns('__pycache__'))
    (root/'experiments').mkdir()
    shutil.copy2(PROJECT/'experiments/records.py', root/'experiments/records.py')
    for name in ENVIRONMENT_FILES:
        shutil.copy2(PROJECT/name, root/name)
    frozen = tmp_path/'frozen'
    inventory = freeze(root, frozen)
    assert 'run_support/records.py' in inventory
    assert 'run_support/source.py' in inventory
    script = "import sys; sys.path.insert(0, sys.argv[1]); from experiments.records import provenance; assert provenance()['code_sha256']"
    subprocess.run([sys.executable, '-I', '-c', script, str(frozen)], cwd=frozen,
                   check=True, capture_output=True, text=True)


def test_migrated_mhd_run_records_only_its_explicit_source_and_reuses(tmp_path):
    from experiments.periodic_mhd.run import run
    from experiments.periodic_mhd.records import load_mhd

    config = {'nx': 12, 'ny': 12, 'T': .01, 'dt': .005, 'snapshots': [0., .01]}
    directory = run(config, root=tmp_path)
    cfg, manifest = load_mhd(directory, require_complete=True)
    source = manifest['source']
    assert source['source_schema'] == 2
    files = source['source_files']
    assert 'run_support/records.py' in files
    assert 'experiments/periodic_mhd/run.py' in files
    assert 'experiments/periodic_mhd/zhou.py' in files
    assert 'experiments/periodic_mhd/analyze.py' not in files
    assert not any(p.startswith('experiments/inverse_cascade/') for p in files)
    assert run(config, root=tmp_path) == directory
    assert cfg['T'] == .01

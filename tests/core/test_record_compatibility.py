"""Legacy source byte encoding, using synthetic files rather than task packages."""
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

"""Saved FEM records at the approved public workflow seam, using real six-scheme solves."""
import hashlib
import importlib
import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from experiments.fem_ns.analyze import analyze
from experiments.fem_ns.problems import build_model
from experiments.fem_ns.reconstruct import reconstruct
from experiments.fem_ns.run import effective_config, load_fem, run
from experiments.workflow import digest, write_json
from solver.integrate import integrate
from solver.schemes.registry import SCHEMES, make_scheme, scheme_info


def test_saved_record_can_be_loaded_without_computation_entry_points(tmp_path):
    directory = run({'nx': 4, 'ny': 3, 'T': .01, 'dt': .01}, root=tmp_path)
    # A fresh reader process must not need any experiment runner or optional
    # computation backend, even if a prior process created a PETSc-capable run.
    script = '''
import importlib.abc
from pathlib import Path
import sys
class ReadonlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'experiments.workflow' or fullname == 'petsc4py' or (
            fullname.startswith('experiments.') and fullname.endswith('.run')):
            raise AssertionError('Reader imported computation entry point: '+fullname)
sys.meta_path.insert(0, ReadonlyImports())
from experiments.fem_ns.records import load_fem
cfg, manifest = load_fem(Path(sys.argv[1]), require_complete=True)
assert cfg['nx'] == 4 and manifest['status'] == 'complete'
'''
    subprocess.run([sys.executable, '-c', script, str(directory)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('name', SCHEMES)
@pytest.mark.parametrize('schedule', [{'dt': .01}, {'steps': [.007, .013]}], ids=['fixed', 'given'])
def test_six_scheme_saved_fields_pressure_edges_and_readonly_analysis(tmp_path, monkeypatch, name, schedule):
    config = {'nx': 4, 'ny': 3, 'T': .02, 'scheme': name,
              'snapshots': [0., .0035, .02], 'stokes_ordering': 'metis', **schedule}
    directory = run(config, root=tmp_path)
    cfg, manifest = load_fem(directory, require_complete=True)
    assert manifest['source']['packages']['scikit-fem'].startswith('12.')
    assert cfg['stokes_ordering'] == 'metis'
    model, initial = build_model(cfg)
    expected = integrate(model, make_scheme(name), initial, cfg['actual_steps'], snapshots=cfg['snapshots'])
    points = np.array([[.23, 1.31, .67], [.17, .62, .43]])
    fields = reconstruct(directory, points)
    np.testing.assert_allclose(fields['velocity'], model.evaluate_velocity(model.vector(expected.final), points), atol=1e-14)
    np.testing.assert_allclose(fields['stage_pressure'], np.stack([
        model.evaluate_pressure(stage.pressure, points) for stage in expected.final_stages]), atol=1e-13)
    assert np.max(np.abs(fields['velocity'])) > 1e-4
    for i, state in enumerate(expected.snapshots):
        snapshot = reconstruct(directory, points, snapshot=i)
        np.testing.assert_allclose(snapshot['velocity'], model.evaluate_velocity(model.vector(state), points), atol=1e-14)
        assert snapshot['stage_pressure'] is None
        assert snapshot['time'] == state.t and snapshot['r'] == state.r
    info = scheme_info(name)
    with h5py.File(directory/'results.h5', 'r') as data:
        assert data['final/u'].shape[0] > data['mesh/coordinates'].shape[1]
        assert data['roots/candidates'].shape == (2, info.n_stages, info.n_roots)
        assert np.max(data['stages/residual'][:]) < 1e-9
        assert np.max(data['stages/continuity_residual'][:]) < 1e-12
        assert np.max(data['stages/scalar_residual'][:]) < 1e-10
        for values in data['final/stage_pressure'][:]:
            assert abs(model.pressure_integral(values)) < 1e-12
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    # Readonly consumer guard: fail if any numerical construction/driver is invoked.
    def forbidden(*args, **kwargs):
        raise AssertionError('analysis launched computation')
    monkeypatch.setattr(importlib.import_module('experiments.fem_ns.run'), 'build_model', forbidden)
    monkeypatch.setattr(importlib.import_module('experiments.fem_ns.run'), 'integrate', forbidden)
    monkeypatch.setattr(type(model), '__init__', forbidden)
    monkeypatch.setattr(type(model.stokes), '__init__', forbidden)
    report = analyze(directory, root=tmp_path)
    summary = json.loads((report/'analysis.json').read_text())
    assert summary['status'] == 'complete' and summary['accepted_steps'] == 2
    assert summary['final_time'] == .02
    assert (report/'fields.csv').exists() and (report/'history.csv').exists()
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize('contained', [True, False], ids=['contained', 'external'])
def test_analysis_run_locator_is_relative_only_for_contained_runs(tmp_path, contained):
    root = tmp_path/'project'
    directory = run({'nx': 2, 'ny': 2, 'T': .01}, root=root if contained else tmp_path/'external')
    summary = json.loads((analyze(directory, root=root)/'analysis.json').read_text())
    expected = f'runs/fem_ns/{directory.name}' if contained else str(directory.resolve())
    assert summary['run'] == expected
    assert (root/summary['run']).resolve() == directory.resolve()


def refresh_checksum(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['results_sha256'] = digest(directory/'results.h5')
    write_json(directory/'manifest.json', manifest)


def test_reconstructs_known_quadratic_edge_field_and_linear_pressure(tmp_path):
    directory = run({'nx': 2, 'ny': 2, 'T': .01}, root=tmp_path)
    # A synthetic full-P2 record fixture: one edge basis 4*lambda_i*lambda_j,
    # supported on two triangles; a vertex-only export would give zero everywhere.
    with h5py.File(directory/'results.h5', 'a') as data:
        locations = data['dofs/velocity_locations'][:, data['dofs/component_u'][:]]
        edge = np.all(locations == [[.5], [.5]], axis=0).astype(float)
        data['final/u'][:] = edge
        data['final/v'][:] = 0.
        p = data['dofs/pressure_locations'][:]
        data['final/stage_pressure'][:] = p[0]+2*p[1]-2.
    refresh_checksum(directory)
    fields = reconstruct(directory, np.array([[.25], [.4]]))
    np.testing.assert_allclose(fields['velocity'][:, 0], [.55, 0.], atol=2e-15)
    np.testing.assert_allclose(fields['stage_pressure'][:, 0], -.95, atol=2e-15)
    walls = reconstruct(directory, np.array([[0., .3, 2., .7], [.4, 0., .3, 1.]]))
    np.testing.assert_allclose(walls['velocity'], 0., atol=2e-15)


def test_shortened_final_step_empty_snapshots_and_reconstruction_input_validation(tmp_path):
    directory = run({'nx': 3, 'ny': 2, 'T': .025, 'dt': .01, 'snapshots': []}, root=tmp_path)
    cfg, _ = load_fem(directory, require_complete=True)
    np.testing.assert_allclose(cfg['actual_steps'], [.01, .01, .005])
    assert reconstruct(directory, np.array([[.3], [.4]]))['time'] == .025
    for points in (np.zeros((3, 2)), np.array([[np.nan], [.2]]), np.array([[2.1], [.2]])):
        with pytest.raises(ValueError, match='points'):
            reconstruct(directory, points)
    with pytest.raises(ValueError, match='snapshot'):
        reconstruct(directory, np.array([[.3], [.4]]), snapshot=0)
    assert json.loads((analyze(directory, root=tmp_path)/'analysis.json').read_text())['accepted_steps'] == 3


@pytest.mark.parametrize('fail', [False, True], ids=['complete', 'accepted-prefix'])
def test_snapshot_mapping_uses_compensated_schedule_and_earlier_ties(tmp_path, monkeypatch, fail):
    runner = importlib.import_module('experiments.fem_ns.run')
    if fail:
        real_build = runner.build_model
        def with_bad_external_force(cfg):
            model, initial = real_build(cfg)
            external_force = model.force
            model.force = lambda t: external_force(t) if t < 9.985 else np.full_like(model.zero_velocity(), np.nan)
            return model, initial
        monkeypatch.setattr(runner, 'build_model', with_bad_external_force)
    steps = np.full(1000, .01)
    steps[100::100] = np.nextafter(.01, np.inf)
    requests = [10., 9.995, np.nextafter(9.995, np.inf), 0., 9.995]
    directory = run({'nx': 2, 'ny': 2, 'T': 10., 'steps': steps.tolist(), 'scheme': 'imex_euler',
                     'snapshots': requests}, root=tmp_path)
    _, manifest = load_fem(directory, require_complete=not fail)
    assert manifest['status'] == ('failed' if fail else 'complete')
    with h5py.File(directory/'results.h5', 'r') as data:
        indices = [999, 999, 0, 999] if fail else [1000, 999, 999, 0, 999]
        assert len(data['dt']) == (999 if fail else 1000)
        np.testing.assert_array_equal(data['snapshots/requested_times'][:], requests[1:] if fail else requests)
        np.testing.assert_array_equal(data['snapshots/actual_times'][:], data['times'][:][indices])
    fields = reconstruct(directory, np.array([[.3], [.4]]), snapshot=0 if fail else 1)
    assert fields['time'] == pytest.approx(9.99, abs=1e-12)


def test_identity_reuse_rerun_and_checksum_gate(tmp_path, monkeypatch):
    config = {'nx': 3, 'ny': 2, 'T': .01}
    directory = run(config, root=tmp_path)
    assert load_fem(directory)[1]['rerun'] is False
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert run(config, root=tmp_path) == directory
    repeated = run(config, root=tmp_path, rerun=True)
    assert repeated != directory and load_fem(repeated)[1]['rerun'] is True
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    monkeypatch.setenv('NUMEXPR_NUM_THREADS', '2')
    assert run(config, root=tmp_path) != directory
    monkeypatch.undo()
    with pytest.raises(ValueError, match='separate'):
        analyze(directory, root=directory)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    with h5py.File(directory/'results.h5', 'a') as data:
        data['final/u'][0] += 1.
    with pytest.raises(ValueError, match='checksum'):
        analyze(directory, root=tmp_path)
    assert run(config, root=tmp_path) != directory


@pytest.mark.parametrize('change', ['missing_mesh', 'missing_map', 'connectivity', 'component_map',
    'map_dtype', 'edge_coefficients', 'nonfinite', 'boundary', 'pressure', 'gauge', 'library',
    'element', 'layout', 'schema', 'pressure_semantics', 'times', 'diagnostics', 'snapshots',
    'roots', 'root_count', 'root_padding', 'status'])
def test_reader_rejects_incomplete_or_incompatible_representation_despite_checksum(tmp_path, change):
    config = {'nx': 3, 'ny': 2, 'T': .01}
    directory = run(config, root=tmp_path)
    with h5py.File(directory/'results.h5', 'a') as data:
        if change == 'missing_mesh': del data['mesh/coordinates']
        elif change == 'missing_map': del data['dofs/free']
        elif change == 'connectivity': data['mesh/connectivity'][0, 0] = 99
        elif change == 'component_map': data['dofs/component_u'][0] = 1
        elif change == 'map_dtype':
            values = data['dofs/free'][:].astype(float)
            del data['dofs/free']
            data['dofs/free'] = values
        elif change == 'edge_coefficients':
            del data['final/u']
            data['final/u'] = np.zeros(data['mesh/coordinates'].shape[1])
        elif change == 'nonfinite': data['final/u'][0] = np.nan
        elif change == 'boundary': data['final/u'][0] = 1.
        elif change == 'pressure': del data['final/stage_pressure']
        elif change == 'gauge': data['final/stage_pressure'][:] += 1.
        elif change == 'library': data.attrs['scikit_fem_version'] = '99.0'
        elif change == 'element': data.attrs['velocity_order'] = 1
        elif change == 'layout': data.attrs['field_layout'] = 'staggered_mac'
        elif change == 'schema': data.attrs['schema_version'] = 2
        elif change == 'pressure_semantics': data.attrs['pressure_semantics'] = 'physical_endpoint'
        elif change == 'times': data['times'][-1] = 0.
        elif change == 'diagnostics': del data['diagnostics/divergence_l2']
        elif change == 'snapshots': data['snapshots/actual_times'][-1] = .001
        elif change == 'roots': data['roots/candidates'][0, 0, 0] = np.nan
        elif change == 'root_count': data['roots/count'][0, 0] = 99
        elif change == 'root_padding':
            count = data['roots/count'][0, 0]
            assert count < 3
            data['roots/candidates'][0, 0, count] = 0.
        elif change == 'status': data.attrs['status'] = 'running'
    refresh_checksum(directory)
    with pytest.raises(ValueError):
        load_fem(directory)
    assert run(config, root=tmp_path) != directory


@pytest.mark.parametrize('missing', ['code_sha256', 'python', 'platform', 'threads'])
def test_reader_requires_saved_provenance_even_when_identity_is_refreshed(tmp_path, missing):
    directory = run({'nx': 3, 'ny': 2, 'T': .01}, root=tmp_path)
    cfg, manifest = load_fem(directory)
    del manifest['source'][missing]
    manifest['identity'] = hashlib.sha256(json.dumps({'config': cfg, 'source': manifest['source']},
                                                     sort_keys=True).encode()).hexdigest()
    write_json(directory/'manifest.json', manifest)
    with pytest.raises(ValueError, match='provenance'):
        load_fem(directory)


def test_failed_integrations_preserve_only_accepted_prefix_and_are_not_reused(tmp_path, monkeypatch):
    runner = importlib.import_module('experiments.fem_ns.run')
    config = {'nx': 3, 'ny': 2, 'T': .03, 'steps': [.01, .02], 'scheme': 'imex_euler_mrsav',
              'snapshots': [0., .005, .01, .03]}
    real_build = runner.build_model
    for fail_at, accepted in [(.005, 1), (-1., 0)]:
        def with_bad_external_force(cfg):
            model, initial = real_build(cfg)
            external_force = model.force
            model.force = lambda t: external_force(t) if t < fail_at else np.full_like(model.zero_velocity(), np.nan)
            return model, initial
        monkeypatch.setattr(runner, 'build_model', with_bad_external_force)
        directory = run(config, root=tmp_path)
        _, manifest = load_fem(directory)
        assert manifest['status'] == 'failed' and manifest['error']
        with pytest.raises(ValueError, match='Complete'):
            load_fem(directory, require_complete=True)
        assert run(config, root=tmp_path) != directory
        summary = json.loads((analyze(directory, root=tmp_path)/'analysis.json').read_text())
        assert summary['status'] == 'failed' and summary['accepted_steps'] == accepted
        assert summary['final_time'] == (.01 if accepted else 0.)
        fields = reconstruct(directory, np.array([[.3], [.4]]))
        assert (fields['stage_pressure'] is None) == (accepted == 0)
        with h5py.File(directory/'results.h5', 'r') as data:
            np.testing.assert_array_equal(data['snapshots/actual_times'][:], [0., 0., .01] if accepted else [0., 0.])


@pytest.mark.parametrize('config', [{'steps': [0., .02]}, {'steps': [.01]}, {'dt': -.1},
    {'steps': [.01, .01], 'dt': .01}, {'nx': 1, 'ny': 1}, {'quadrature_order': 4},
    {'scheme': 'unknown'}, {'r0': .1}, {'snapshots': [np.nan]}, {'boundary': 'periodic'},
    {'nu': np.inf}, {'cache_size': False}, {'gamma': -1.}, {'stokes_ordering': 'unknown'},
    {'stokes_factor_backend': 'unknown'}, {'stokes_threads': 4},
    {'stokes_factor_backend': 'pardiso', 'stokes_threads': True},
    {'stokes_factor_backend': 'pardiso', 'stokes_ordering': 'metis'}])
def test_invalid_fem_config_is_rejected_before_writing(tmp_path, config):
    with pytest.raises(ValueError):
        run(config, root=tmp_path)
    assert not (tmp_path/'runs').exists()

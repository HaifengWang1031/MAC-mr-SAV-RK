"""Cylinder records at public run/load/reconstruct/readonly-analysis seams."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from solver.schemes.registry import SCHEMES, scheme_info
from solver.fem import FEMModel
from PIL import Image, GifImagePlugin

from experiments.cylinder_ns.run import run, load_cylinder
from experiments.cylinder_ns.reconstruct import reconstruct
from experiments.cylinder_ns.analyze import analyze
from experiments.workflow import digest, write_json


def test_cylinder_fixed_run_reconstruction_analysis_reuse_and_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = {'resolution': .25, 'T': .01, 'mode': 'fixed', 'dt': .005,
              'scheme': 'sdirk3_mrsav', 'snapshots': [0., .01]}
    directory = run(config, root=tmp_path)
    cfg, manifest = load_cylinder(directory, require_complete=True)
    assert manifest['status'] == 'complete' and cfg['reynolds'] == 100.
    assert cfg['pressure_constraint'] == 'natural_outflow'
    points = np.array([[0., .4, 1.8], [.2, .2, .18]])
    fields = reconstruct(directory, points)
    assert fields['time'] == .01 and np.isfinite(fields['velocity']).all()
    assert fields['velocity'][0, 0] == pytest.approx(6*.2*(.41-.2)/.41**2, abs=2e-13)
    assert np.isfinite(fields['physical_pressure']).all()
    with h5py.File(directory/'results.h5', 'r') as data:
        assert data['observations/time'].shape == (3,)
        assert data['final/physical_pressure'].shape[0] == data['dofs/pressure_locations'].shape[1]
        assert np.max(data['stages/continuity_residual'][:]) < 1e-9
    before = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Readonly analysis must not construct a dynamical model')

    with monkeypatch.context() as guard:
        guard.setattr(FEMModel, '__init__', forbidden)
        report = analyze(directory, root=tmp_path)
    assert (report/'analysis.json').is_file()
    assert (report/'figures/mesh.png').is_file() and (report/'figures/velocity.png').is_file()
    assert (report/'figures/vorticity.png').is_file() and (report/'figures/forces.png').is_file()
    assert before == {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    assert run(config, root=tmp_path) == directory
    assert run(config, root=tmp_path, rerun=True) != directory
    with pytest.raises(ValueError):
        reconstruct(directory, np.array([[.2], [.2]]))


@pytest.mark.parametrize('scheme', SCHEMES)
def test_every_fixed_format_roundtrips_natural_pressure(scheme: str, tmp_path: Path):
    directory = run({'resolution': .25, 'T': .004, 'mode': 'fixed', 'dt': .002,
                     'scheme': scheme, 'snapshots': []}, root=tmp_path)
    cfg, _ = load_cylinder(directory, require_complete=True)
    info = scheme_info(scheme)
    assert cfg['pressure_semantics'].endswith('natural_outflow')
    with h5py.File(directory/'results.h5', 'r') as data:
        assert data['stages/residual'].shape == (2, info.n_stages)
        assert data['final/stage_pressure'].shape[0] == info.n_stages
        assert np.isfinite(data['final/physical_pressure'][:]).all()


@pytest.mark.parametrize('backend', ['direct', 'recycled'])
@pytest.mark.parametrize('kind', ['I', 'PI'])
@pytest.mark.parametrize('scheme', ['sdirk2', 'sdirk2_mrsav', 'sdirk3', 'sdirk3_mrsav'])
def test_adaptive_attempts_exact_endpoint_and_actual_snapshot_mapping(kind: str, scheme: str, backend: str, tmp_path: Path):
    directory = run({'resolution': .25, 'T': .02, 'mode': 'adaptive', 'initial_step': .02,
                     'scheme': scheme, 'stokes_backend': backend,
                     'scalar_atol': 1e-5 if scheme.endswith('_mrsav') else None,
                     'controller': {'kind': kind, 'rtol': 1e-4, 'max_step': .02},
                     'snapshots': [0., .011, .02]}, root=tmp_path)
    cfg, manifest = load_cylinder(directory, require_complete=True)
    with h5py.File(directory/'results.h5', 'r') as data:
        times = data['times'][:]
        accepted = data['attempts/accepted'][:]
        assert times[-1] == cfg['T'] and np.sum(accepted) == len(times)-1
        assert np.all(data['attempts/error'][:][accepted] <= 1.)
        if scheme.endswith('_mrsav'):
            assert np.any(~accepted)
        actual = [times[int(np.argmin(abs(times-t)))] for t in cfg['snapshots']]
        np.testing.assert_array_equal(data['snapshots/actual_times'][:], actual)
        assert len(data['observations/time']) == len(times)
    assert manifest['metrics']['integration_cpu_seconds'] >= 0.


def test_adaptive_failed_accepted_prefix_remains_readable(tmp_path: Path):
    directory = run({'resolution': .25, 'T': .004, 'mode': 'adaptive', 'initial_step': .001,
                     'controller': {'kind': 'I', 'max_steps': 1}, 'snapshots': [0., .004]}, root=tmp_path)
    _, manifest = load_cylinder(directory)
    assert manifest['status'] == 'failed' and manifest['metrics']['accepted_steps'] == 1
    with pytest.raises(ValueError):
        load_cylinder(directory, require_complete=True)
    fields = reconstruct(directory, np.array([[.4], [.2]]))
    assert fields['time'] == .001 and np.isfinite(fields['physical_pressure']).all()


@pytest.mark.parametrize('change', [{'resolution': True}, {'circle_segments': 18},
                                    {'quadrature_order': True}, {'r0': float('nan')},
                                    {'scheme': 'imex_euler', 'mode': 'adaptive'}])
def test_invalid_cylinder_inputs_do_not_create_runs(change: dict, tmp_path: Path):
    with pytest.raises(ValueError):
        run({'resolution': .25, 'T': .004, **change}, root=tmp_path)
    assert not (tmp_path/'runs').exists()


@pytest.mark.parametrize('corruption', ['physical_label', 'trace', 'circle_geometry', 'accepted_flag'])
def test_semantic_corruption_is_rejected_even_with_updated_checksum(corruption: str, tmp_path: Path):
    directory = run({'resolution': .25, 'T': .004, 'initial_step': .001}, root=tmp_path)
    with h5py.File(directory/'results.h5', 'a') as data:
        if corruption == 'physical_label':
            data.attrs['physical_pressure_semantics'] = 'incremental_stage_multiplier_natural_outflow'
        elif corruption == 'trace':
            values = data['dofs/dirichlet_values'][:]
            index = np.flatnonzero(values)[0]
            data['dofs/dirichlet_values'][index] += .01
        elif corruption == 'circle_geometry':
            xy = data['mesh/coordinates'][:]
            index = np.flatnonzero(abs(np.linalg.norm(xy-np.array([[.2], [.2]]), axis=0)-.05) < 1e-10)[0]
            data['mesh/coordinates'][0, index] += .001
        else:
            data['attempts/accepted'][0] = False
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['results_sha256'] = digest(directory/'results.h5')
    write_json(directory/'manifest.json', manifest)
    with pytest.raises(ValueError):
        load_cylinder(directory)


def test_saved_field_animation_has_actual_frames(tmp_path: Path):
    directory = run({'resolution': .25, 'T': .006, 'mode': 'fixed', 'dt': .002,
                     'snapshots': [0., .002, .004, .006]}, root=tmp_path)
    report = analyze(directory, root=tmp_path)
    with Image.open(report/'figures/wake.gif') as image:
        assert isinstance(image, GifImagePlugin.GifImageFile)
        assert image.n_frames == 4 and image.size == (1200, 500)
    with h5py.File(directory/'results.h5', 'r') as data:
        assert np.max(abs(data['snapshots/u'][-1]-data['snapshots/u'][0])) > 1e-4

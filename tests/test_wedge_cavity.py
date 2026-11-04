"""Moving-lid FEM trace, SAV lifting work and recorded wedge startup."""
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.wedge_cavity.config import effective_config, validate_stored
from experiments.wedge_cavity.run import build, run
from experiments.wedge_cavity.fields import streamline_levels
from solver.fem.cavity import ClosedCavityModel
from solver.fem.wedge import WedgeSpace


def test_contours_retain_the_weak_opposite_cell_instead_of_only_the_main_sign():
    levels = streamline_levels(np.array([-.1, -.02, 0., 1e-4]))
    assert np.any((levels > 0.) & (levels < 1e-4))
    assert np.any((levels < 0.) & (levels > -.1))
    assert np.all(np.diff(levels) > 0.)
    assert len(streamline_levels(np.zeros(5))) == 0


def test_stationary_inclined_walls_include_both_lid_corners():
    model = ClosedCavityModel(WedgeSpace(apex_angle=25., lid_cells=12), .01)
    mesh, basis = model.velocity_basis.mesh, model.velocity_basis
    walls = np.concatenate([mesh.boundaries[key] for key in ('wall_left', 'wall_right')])
    fixed = basis.get_dofs(facets=walls).all()
    np.testing.assert_array_equal(model._boundary_values[fixed], 0.)
    full = model.full_velocity(model.zero_velocity())
    u_dofs = model.component_dofs[0]
    x, y = basis.doflocs[:, u_dofs]
    moving = (np.abs(y) < 1e-14) & (np.abs(x) < .5-1e-14)
    np.testing.assert_array_equal(full[u_dofs[moving]], 1.)
    np.testing.assert_array_equal(full[model.component_dofs[1]], 0.)


def test_lifting_work_matches_independent_skew_form_and_cancels_free_pairing():
    model = ClosedCavityModel(WedgeSpace(lid_cells=12), .01)
    velocity = np.random.default_rng(53).normal(scale=.15, size=len(model.free_dofs))
    full = model.full_velocity(velocity)
    physical = model.velocity_basis.interpolate(full)
    lifting = model.velocity_basis.interpolate(model._boundary_values)
    advected = np.einsum('ijeq,jeq->ieq', physical.grad, physical)
    test_advected = np.einsum('ijeq,jeq->ieq', lifting.grad, physical)
    reference = .5*float(np.sum((np.sum(advected*lifting, axis=0)
                                - np.sum(test_advected*physical, axis=0))*model.velocity_basis.dx))
    nonlinear, work = model.nonlinear_with_lifting(velocity)
    assert abs(reference) > 1e-6
    np.testing.assert_allclose(work, reference, atol=2e-14)
    np.testing.assert_allclose(model.inner(nonlinear, velocity)+work, 0., atol=2e-13)


def test_startup_records_zero_sav_initial_value_exact_endpoint_and_final_snapshot(tmp_path: Path):
    config = {'lid_cells': 8, 'T': .03, 'dt': .01, 'snapshot_every': .02, 'log_every': 1}
    directory = run(config, root=tmp_path)
    manifest = json.loads((directory/'manifest.json').read_text())
    cfg = json.loads((directory/'config.json').read_text())
    assert manifest['status'] == 'complete' and manifest['final_time'] == .03
    assert cfg['r0'] == 0. and cfg['lid_corner_policy'] == 'stationary'
    with h5py.File(directory/'results.h5', 'r') as data:
        assert data['diagnostics/r'][0] == 0.
        np.testing.assert_array_equal(data['snapshots/times'][:], [0., .02, .03])
        assert np.max(data['stage_residual'][:, 0]) < 1e-9
        assert np.max(data['stage_residual'][:, 3]) < 1e-11
        assert data.attrs['scalar_solver'] == 'fixed_count_frozen_factor'
        assert np.all(data['corrections/count'][:] == 1)
        _, _, model = build(config)
        final = model.state(.03, model.zero_velocity())
        final = type(final)(.03, data['snapshots/u'][-1], data['snapshots/v'][-1],
                            float(data['snapshots/r'][-1]))
        assert model.max_abs(model.apply_D(model.vector(final))) < 1e-11


def test_legacy_configuration_keeps_the_actual_old_sav_and_corner_conventions():
    legacy = effective_config({'lid_cells': 8, 'T': .02, 'dt': .01})
    legacy.pop('r0', None)
    legacy.pop('lid_corner_policy', None)
    restored = validate_stored(legacy)
    assert restored['r0'] == 1. and restored['lid_corner_policy'] == 'lid'


@pytest.mark.skipif(any(shutil.which(tool) is None for tool in ('ffmpeg', 'xelatex', 'pdffonts')),
                    reason='optional animation tools unavailable')
@pytest.mark.parametrize('detail', ['vorticity', 'lower_cell'])
def test_movie_reads_a_saved_run_and_keeps_its_checksum_and_endpoint(tmp_path: Path, monkeypatch, detail: str):
    from experiments.wedge_cavity.report import render_frames
    from experiments.workflow import digest

    directory = run({'lid_cells': 8, 'T': .02, 'dt': .01,
                     'snapshot_every': .01, 'log_every': 1}, root=tmp_path)
    before = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    monkeypatch.setattr(ClosedCavityModel, 'solve_columns', lambda *a, **kw:
        (_ for _ in ()).throw(AssertionError('Animation launched a flow solve')))
    output = tmp_path/'reports'/'startup.mp4'
    # A nonintegral figure-size*dpi used to make PGF and Agg differ by one pixel.
    record = render_frames(directory, output=output, fps=5, dpi=61, step=2, hold=0, detail=detail)
    assert record['status'] == 'complete' and record['font_verified']
    assert record['actual_times'] == [0., .02] and record['inputs_unchanged']
    assert output.stat().st_size > 1000 and output.with_suffix('.gif').stat().st_size > 1000
    assert before == {p.name: digest(p) for p in directory.iterdir() if p.is_file()}

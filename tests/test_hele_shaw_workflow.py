"""Actual run/load/readonly analysis, exact endpoints and independent reuse identity."""
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

pytest.importorskip('gmsh')

from experiments.hele_shaw_airfoil.records import load
from experiments.hele_shaw_airfoil.run import run
from solver.fem.hele_shaw import HeleShawModel

SMALL = dict(gap_ratio=.3, layer_cells=1, surface_points=32, far_size=.6,
             T=.013, dt=.005, snapshots=[0., .005, .013], log_every=1)


@pytest.mark.parametrize('adaptive', [False, True])
def test_public_saved_record_and_endpoint(tmp_path: Path, adaptive: bool):
    config = SMALL.copy()
    if adaptive:
        config.update(mode='adaptive', dt=None, initial_step=.001, scalar_atol=1e-4,
                      controller={'kind': 'PI', 'atol': 1e-8, 'rtol': 1e-3})
    directory = run(config, root=tmp_path)
    cfg, manifest, mesh, vb, pb = load(directory)
    assert manifest['status'] == 'complete' and manifest['final_time'] == cfg['T']
    assert cfg['hele_shaw_schema_version'] == 1
    assert 'mean_zero' not in cfg['pressure_semantics']
    with h5py.File(directory/'results.h5', 'r') as data:
        assert data['times'][-1] == .013
        assert data['snapshots/actual_times'][-1] == .013
        assert data['snapshots/physical_pressure'].shape == (3, pb.N)
        assert np.max(data['stages/continuity_residual'][:]) < 1e-10
        assert data['darcy'].attrs['residual'] < 1e-9
        if adaptive:
            assert 'attempts' in data
    assert run(config, root=tmp_path) == directory
    if not adaptive:
        other = run({**config, 'angle': -10.}, root=tmp_path)
        assert other != directory
        from experiments.hele_shaw_airfoil.campaign import compare
        with pytest.raises(ValueError, match='matching physical'):
            compare(directory, other)


def test_roundoff_endpoint_is_exact_even_when_three_times_dt_is_not_T(tmp_path: Path):
    directory = run({**SMALL, 'T': .9, 'dt': .3, 'snapshots': [0., .9],
                     'convection': False, 'pressure_drop': .01}, root=tmp_path)
    cfg, manifest, *_ = load(directory)
    assert manifest['status'] == 'complete' and manifest['final_time'] == .9
    with h5py.File(directory/'results.h5', 'r') as data:
        assert data['times'][-1] == .9 and data['snapshots/actual_times'][-1] == .9


@pytest.mark.skipif(any(shutil.which(x) is None for x in ['xelatex', 'ffmpeg']), reason='optional plotting tools')
def test_analysis_uses_saved_fields_and_cannot_run_a_flow_or_mesh(tmp_path: Path, monkeypatch):
    from experiments.hele_shaw_airfoil.analyze import analyze
    import gmsh
    directory = run(SMALL, root=tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Readonly analysis started a computation')
    monkeypatch.setattr(HeleShawModel, 'solve_columns', forbidden)
    monkeypatch.setattr(gmsh.model.mesh, 'generate', forbidden)
    report = tmp_path/'reports'
    summary = analyze(directory, output=report, movie=True, dpi=65, fps=5)
    assert summary['inputs_unchanged'] and summary['actual_times'] == [0., .005, .013]
    assert (report/'final.png').stat().st_size > 1000
    assert (report/'evolution.mp4').stat().st_size > 1000
    with pytest.raises(ValueError, match='Separate'):
        analyze(directory, output=directory/'report')

"""Wake grading, optional-record compatibility and independent rotation diagnostics."""
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.hele_shaw_airfoil.config import effective_config
from experiments.hele_shaw_airfoil.wake_analysis import flow_indicators
from experiments.hele_shaw_airfoil.wake_campaign import configurations


def test_wake_campaign_preserves_speed_geometry_and_sav_controls():
    cases = configurations()
    for name in ('re10000', 're30000'):
        cfg = effective_config(cases[name])
        speed = cfg['gap']**2*cfg['pressure_drop']/(12*cfg['rho']*cfg['nu']*9*cfg['chord'])
        assert speed == pytest.approx(1.)
        assert speed*cfg['chord']/cfg['nu'] == pytest.approx(int(name[2:]))
        assert cfg['scheme'] == 'sdirk3_mrsav' and cfg['r0'] == 0.
        assert cfg['T'] == 20. and cfg['snapshots'][-1] == 20.
        assert cfg['angle'] == 10. and cfg['gap_ratio'] == .02
        assert cfg['outlet_treatment'] == 'natural'
    assert cases['pilot_time']['controller']['max_step'] == .5*cases['pilot_high']['controller']['max_step']


def test_vorticity_alone_does_not_classify_shear_as_rotation():
    weights = np.ones(4)/4
    u = np.array([[1., 1., -1., -1.], [0., 0., 0., 0.]])
    grad = np.zeros((2, 2, 4));grad[0, 1] = 3.
    shear = flow_indicators(u, grad, weights, 1.)
    assert shear['enstrophy'] == pytest.approx(9.)
    assert shear['maximum_swirling_strength'] == 0.
    assert shear['rotating_area'] == 0.
    grad[0, 1], grad[1, 0] = -2., 2.
    rotation = flow_indicators(u, grad, weights, 1.)
    assert rotation['maximum_swirling_strength'] == pytest.approx(2.)
    assert rotation['rotating_area'] == pytest.approx(1.)
    assert rotation['backflow_area'] == pytest.approx(.5)


def test_new_options_are_explicit_and_validate_pairs_without_changing_old_configs():
    old = effective_config({})
    assert all(k not in old for k in ('wake_region', 'wake_size', 'cache_size'))
    with pytest.raises(ValueError, match='together'):
        effective_config({'wake_size': .03})
    with pytest.raises(ValueError, match='inside'):
        effective_config({'wake_region': [1., 8., -1., 1.], 'wake_size': .03})
    new = effective_config({'wake_region': [1., 5., -.7, .5], 'wake_size': .03, 'cache_size': 1})
    assert new['cache_size'] == 1


def test_gmsh_wake_box_refines_downstream_elements():
    pytest.importorskip('gmsh')
    from solver.fem.airfoil import make_airfoil_mesh
    areas = []
    for wake in [False, True]:
        kwargs = dict(wake_region=(1.2, 3., -.4, .4), wake_size=.09) if wake else {}
        mesh = make_airfoil_mesh(gap_ratio=.3, layer_cells=1, surface_points=32, far_size=.6, **kwargs)
        p = mesh.p[:, mesh.t];center = p.mean(axis=1)
        area = .5*abs((p[0, 1]-p[0, 0])*(p[1, 2]-p[1, 0])-(p[1, 1]-p[1, 0])*(p[0, 2]-p[0, 0]))
        selected = (center[0]>1.5)&(center[0]<2.5)&(abs(center[1])<.3)
        areas.append(float(np.median(area[selected])))
    assert areas[1] < .4*areas[0]


@pytest.mark.skipif(any(shutil.which(x) is None for x in ['xelatex', 'ffmpeg']), reason='optional rendering tools')
def test_wake_analysis_uses_saved_fields_and_preserves_the_run(tmp_path: Path, monkeypatch):
    gmsh = pytest.importorskip('gmsh')
    from experiments.hele_shaw_airfoil.run import run
    from experiments.hele_shaw_airfoil.records import load
    from experiments.hele_shaw_airfoil.wake_analysis import analyze_wake
    from solver.fem.hele_shaw import HeleShawModel
    cfg = dict(gap_ratio=.3, layer_cells=1, surface_points=32, far_size=.6,
               T=.006, dt=.002, snapshots=[0., .006], cache_size=1,
               wake_region=[1.2, 4., -.5, .5], wake_size=.15)
    directory = run(cfg, root=tmp_path)
    saved, *_ = load(directory)
    assert saved['wake_region'] == cfg['wake_region']
    def forbidden(*a, **k):raise AssertionError('Analysis started a flow or remeshed')
    monkeypatch.setattr(HeleShawModel, 'solve_columns', forbidden)
    monkeypatch.setattr(gmsh.model.mesh, 'generate', forbidden)
    result = analyze_wake(directory, output=tmp_path/'report', dpi=60, fps=5)
    assert result['inputs_unchanged']
    assert (tmp_path/'report/wake.mp4').stat().st_size > 1000
    with h5py.File(directory/'results.h5') as data:
        assert data['times'][-1] == .006

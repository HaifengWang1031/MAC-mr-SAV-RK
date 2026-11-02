"""Independent inlet energy work, zero-r SAV pairing and explicit record identity."""
from pathlib import Path

import h5py
import numpy as np
import pytest
from skfem import MeshTri

from experiments.hele_shaw_airfoil.config import effective_config
from solver.fem.hele_shaw import HeleShawModel
from solver.fem.space import MeshSpace
from solver.schemes.registry import make_scheme


def channel() -> MeshTri:
    return MeshTri.init_tensor(np.linspace(0., 2., 5), np.linspace(-.5, .5, 5)).with_boundaries({
        'inlet': lambda x: np.isclose(x[0], 0.), 'outlet': lambda x: np.isclose(x[0], 2.),
        'walls': lambda x: np.isclose(abs(x[1]), .5)})


def test_inlet_correction_has_known_work_and_preserves_sav_pairing():
    natural = HeleShawModel(MeshSpace(channel()), .1, .4)
    corrected = HeleShawModel(MeshSpace(channel()), .1, .4, inlet_treatment='energy_stable_pressure')
    full = np.zeros(natural.velocity_basis.N)
    for amplitude, ids in zip((4., 2.), natural.component_dofs):
        y = natural.velocity_basis.doflocs[1, ids]
        full[ids] = amplitude*(.25-y*y)
    u = full[natural.free_dofs]
    a, wa = natural.nonlinear_with_lifting(u)
    b, wb = corrected.nonlinear_with_lifting(u)
    # .5 beta *4*(4^2+2^2)*integral[-.5,.5](.25-y^2)^3 dy; integral=1/140.
    expected = 2/7*corrected.beta
    assert corrected.inner(b-a, u) == pytest.approx(expected, abs=3e-13)
    assert wb-wa == pytest.approx(-expected, abs=3e-13)
    assert abs(corrected.inner(b, u)+wb) < 3e-13
    d = corrected.diagnostics(corrected.state(0., u))
    assert d['inlet_kinetic_energy_correction'] == pytest.approx(expected, abs=3e-13)
    assert d['convective_boundary_power'] == pytest.approx(0., abs=3e-13)
    # Outflow at the named inlet gets no correction; wall work remains zero.
    u = -u
    np.testing.assert_array_equal(natural.nonlinear(u), corrected.nonlinear(u))


def test_linear_limit_and_all_sav_stages_keep_original_matrices():
    models = [HeleShawModel(MeshSpace(channel()), .1, .4, convection=False, **kw)
              for kw in ({}, {'inlet_treatment': 'energy_stable_pressure'})]
    states = [m.state(0., m.zero_velocity()) for m in models]
    results = [make_scheme('sdirk3_mrsav').step(m, s, .003) for m, s in zip(models, states)]
    np.testing.assert_array_equal(results[0].state.u, results[1].state.u)
    np.testing.assert_array_equal(results[0].state.v, results[1].state.v)
    assert results[1].state.r == 0.
    assert max(s.continuity_residual for s in results[1].stages) < 1e-11
    combined = HeleShawModel(MeshSpace(channel()), .1, .4,
        inlet_treatment='energy_stable_pressure', outlet_treatment='backflow_damped')
    u = np.random.default_rng(17).normal(size=len(combined.free_dofs))*.1
    n, work = combined.nonlinear_with_lifting(u)
    assert abs(combined.inner(n, u)+work) < 1e-12
    d = combined.diagnostics(combined.state(0., u))
    assert d['convective_boundary_power']+d['inlet_kinetic_energy_correction']+d['outlet_backflow_dissipation'] >= -1e-12


def test_inlet_config_identity_and_saved_public_workflow(tmp_path: Path):
    pytest.importorskip('gmsh')
    from experiments.hele_shaw_airfoil.run import run
    from experiments.hele_shaw_airfoil.records import load
    old = effective_config({})
    assert 'inlet_treatment' not in old
    with pytest.raises(ValueError, match='inlet_treatment'):
        effective_config({'inlet_treatment': 'unknown'})
    cfg = dict(inlet_treatment='energy_stable_pressure', gap_ratio=.3, layer_cells=1,
               surface_points=32, far_size=.6, T=.004, dt=.002, scheme='sdirk3_mrsav')
    new = effective_config(cfg)
    assert new['boundary'] != old['boundary']
    assert 'energy_stable_inlet' in new['pressure_semantics']
    directory = run(cfg, root=tmp_path)
    loaded, manifest, *_ = load(directory)
    assert loaded == new and manifest['status'] == 'complete'
    with h5py.File(directory/'results.h5') as f:
        assert f.attrs['boundary'] == new['boundary']
        assert f.attrs['inlet_treatment'] == 'energy_stable_pressure'
        assert np.isfinite(f['diagnostics/inlet_kinetic_energy_correction'][:]).all()
        assert f['times'][-1] == .004
    import json
    live = json.loads((directory/'progress.json').read_text())
    assert live['status'] == 'complete' and live['t'] == .004
    # Runtime limits stop at a fully accepted state; nothing is clipped.
    stopped = run({**cfg, 'velocity_limit': 1e-12}, root=tmp_path)
    _, failed, *_ = load(stopped, require_complete=False)
    assert failed['status'] == 'failed'
    assert 'Runtime qualification limit exceeded' in failed['error']
    with h5py.File(stopped/'results.h5') as f:
        assert len(f['times']) == 2 and f['times'][-1] == .002
    # A replay starts at a nonzero saved time; the target is an absolute time.
    from tools.probe_hele_shaw_inlet import probe
    adaptive = run({**cfg, 'mode': 'adaptive', 'dt': None, 'T': .006,
                    'initial_step': .002, 'scalar_atol': 1e-4,
                    'controller': {'max_step': .002}}, root=tmp_path)
    replay = probe(adaptive, inlet_treatment='energy_stable_pressure', replay_to=.008)
    assert replay['initial_time'] == .006 and replay['final_time'] == .008
    assert replay['passed']

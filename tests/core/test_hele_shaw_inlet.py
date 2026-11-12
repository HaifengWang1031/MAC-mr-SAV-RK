"""Independent inlet energy work, zero-r SAV pairing and explicit record identity."""
import numpy as np
import pytest
from skfem import MeshTri

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

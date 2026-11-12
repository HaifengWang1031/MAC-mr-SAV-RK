"""Independent boundary work, unchanged positive-outflow solution, and saved labels."""
import numpy as np
import pytest
from skfem import MeshTri

from solver.fem import MeshSpace, OpenFlowModel
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV


def models(trace):
    mesh = MeshTri.init_tensor(np.linspace(0., 2., 5), np.linspace(0., 1., 4)).with_boundaries({
        'inlet': lambda x: np.isclose(x[0], 0.), 'outlet': lambda x: np.isclose(x[0], 2.),
        'walls': lambda x: np.isclose(x[1], 0.) | np.isclose(x[1], 1.)})
    fixed = np.concatenate([mesh.boundaries[key] for key in ('inlet', 'walls')])
    return [OpenFlowModel(MeshSpace(mesh), .1, dirichlet_facets=fixed, boundary_velocity=trace,
                          outlet_treatment=mode) for mode in ('natural', 'backflow_damped')]


def test_backflow_adds_exact_nonnegative_boundary_work_and_keeps_zero_r_pairing():
    natural, damped = models(lambda x, y: (-x*y*(1-y), 0.))
    full = np.zeros(natural.velocity_basis.N)
    indices = natural.component_dofs[0]
    x, y = natural.velocity_basis.doflocs[:, indices]
    full[indices] = -x*y*(1-y)
    u = full[natural.free_dofs]
    a, _ = natural.nonlinear_with_lifting(u)
    b, work = damped.nonlinear_with_lifting(u)
    # At x=2: U=(-2y(1-y),0); .5 integral_0^1 [2y(1-y)]^3 dy = 1/35.
    assert natural.inner(b-a, u) == pytest.approx(1/35, abs=3e-13)
    assert abs(damped.inner(b, u)+work) < 3e-13
    d = damped.diagnostics(damped.state(0., u))
    assert d['outlet_backflow_dissipation'] == pytest.approx(1/35, abs=3e-13)
    # Inflow is prescribed; damping must not be applied to Dirichlet facets.
    assert 'outlet_backflow_dissipation' not in natural.diagnostics(natural.state(0., u))


def test_positive_poiseuille_outflow_has_identical_nonlinearity_and_sav_step():
    natural, damped = models(lambda x, y: (4*y*(1-y), 0.))
    initial = []
    for m in (natural, damped):
        steady = m.solve(m.zero_velocity(), mass=0., viscosity=.1)
        initial.append(m.state(0., steady.velocity))
    np.testing.assert_array_equal(natural.nonlinear(natural.vector(initial[0])),
                                  damped.nonlinear(damped.vector(initial[1])))
    for m, s in zip((natural, damped), initial):
        trial = SDIRK3MRSAV().step(m, s, .01)
        np.testing.assert_allclose(m.vector(trial.state), m.vector(s), atol=3e-11)
        assert abs(trial.state.r) < 1e-10
        _, p = m.instantaneous_solution(m.vector(trial.state), trial.state.t)
        np.testing.assert_allclose(p, .8*(2-m.pressure_basis.doflocs[0]), atol=5e-11)

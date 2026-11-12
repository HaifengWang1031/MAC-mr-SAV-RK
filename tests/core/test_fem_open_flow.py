"""Open-flow Model seam: exact nonzero Poiseuille flow and natural pressure."""
import numpy as np
import pytest
from skfem import MeshTri

from solver.fem import MeshSpace, OpenFlowModel
from solver.schemes.registry import SCHEMES, make_scheme


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('factor_backend', ['superlu', 'pardiso'])
def test_exact_poiseuille_with_natural_outlet_and_fixed_inlet(scheme, factor_backend):
    if factor_backend == 'pardiso':
        pytest.importorskip('pypardiso')
    mesh = MeshTri.init_tensor(np.linspace(0., 2., 5), np.linspace(0., 1., 4)).with_boundaries({
        'inlet': lambda x: np.isclose(x[0], 0.),
        'outlet': lambda x: np.isclose(x[0], 2.),
        'walls': lambda x: np.isclose(x[1], 0.) | np.isclose(x[1], 1.),
    })
    assert mesh.boundaries is not None
    fixed = np.concatenate([mesh.boundaries[name] for name in ('inlet', 'walls')])
    model = OpenFlowModel(MeshSpace(mesh), .1, dirichlet_facets=fixed,
                          boundary_velocity=lambda x, y: (4*y*(1-y), 0.),
                          stokes_ordering='metis' if factor_backend == 'superlu' else 'auto',
                          stokes_factor_backend=factor_backend, stokes_threads=2 if factor_backend == 'pardiso' else None)
    steady = model.solve(model.zero_velocity(), mass=0., viscosity=.1)
    points = np.array([[.1, .6, 1.9, 2.], [.2, .7, .4, .5]])
    expected = np.array([4*points[1]*(1-points[1]), np.zeros(points.shape[1])])
    np.testing.assert_allclose(model.evaluate_velocity(steady.velocity, points), expected, atol=2e-12)
    np.testing.assert_allclose(model.evaluate_pressure(steady.pressure, points), .8*(2-points[0]), atol=3e-12)
    assert model.pressure_integral(steady.pressure) == pytest.approx(1.6, abs=3e-12)
    assert model.max_abs(model.apply_D(steady.velocity)) < 1e-12
    initial = model.state(0., steady.velocity)
    before_u, before_v = initial.u.copy(), initial.v.copy()
    assert model.diagnostics(initial)['kinetic'] == pytest.approx(8/15, abs=2e-12)
    trial = make_scheme(scheme).step(model, initial, .01)
    np.testing.assert_array_equal(initial.u, before_u)
    np.testing.assert_array_equal(initial.v, before_v)
    np.testing.assert_allclose(model.evaluate_velocity(model.vector(trial.state), points), expected, atol=3e-11)
    assert abs(trial.state.r) < 1e-10
    assert all(stage.continuity_residual < 1e-11 for stage in trial.stages)
    _, physical_pressure = model.instantaneous_solution(model.vector(trial.state), trial.state.t)
    np.testing.assert_allclose(model.evaluate_pressure(physical_pressure, points), .8*(2-points[0]), atol=5e-11)


def test_open_convection_has_known_nonzero_boundary_energy_flux():
    mesh = MeshTri.init_tensor(np.linspace(0., 2., 5), np.linspace(0., 1., 4)).with_boundaries({
        'inlet': lambda x: np.isclose(x[0], 0.),
        'outlet': lambda x: np.isclose(x[0], 2.),
        'walls': lambda x: np.isclose(x[1], 0.) | np.isclose(x[1], 1.),
    })
    assert mesh.boundaries is not None
    model = OpenFlowModel(MeshSpace(mesh), .1,
                          dirichlet_facets=np.concatenate([mesh.boundaries[k] for k in ('inlet', 'walls')]),
                          boundary_velocity=lambda x, y: (x, 0.))
    full = np.zeros(model.velocity_basis.N)
    full[model.component_dofs[0]] = model.velocity_basis.doflocs[0, model.component_dofs[0]]
    velocity = full[model.free_dofs]
    nonlinear, work = model.nonlinear_with_lifting(velocity)
    # For U=(x,0), advective + (div U)U/2 = (3x/2,0), not just advective.
    expected = model.project(lambda x, y: (1.5*x, 0.))
    np.testing.assert_allclose(nonlinear, expected, atol=3e-12)
    assert abs(model.inner(nonlinear, velocity)+work) < 2e-13
    diagnostic = model.diagnostics(model.state(0., velocity))
    assert diagnostic['convective_boundary_power'] == pytest.approx(4., abs=2e-13)
    assert diagnostic['boundary_flux'] == pytest.approx(2., abs=2e-13)
    assert diagnostic['continuity_residual'] > .01


def test_physical_pressure_rate_solve_with_known_nonzero_convection():
    mesh = MeshTri.init_tensor(np.linspace(0., 2., 5), np.linspace(0., 1., 4)).with_boundaries({
        'inlet': lambda x: np.isclose(x[0], 0.),
        'outlet': lambda x: np.isclose(x[0], 2.),
        'walls': lambda x: np.isclose(x[1], 0.) | np.isclose(x[1], 1.),
    })
    assert mesh.boundaries is not None
    nu, gradient = .2, .7
    model = OpenFlowModel(MeshSpace(mesh), nu,
                          dirichlet_facets=np.concatenate([mesh.boundaries[k] for k in ('inlet', 'walls')]),
                          boundary_velocity=lambda x, y: (x, -y))
    x, y = model.velocity_basis.doflocs
    full = np.zeros(model.velocity_basis.N)
    for values, indices in zip((x, -y), model.component_dofs):
        full[indices] = values[indices]
    velocity = full[model.free_dofs]
    force = model.project(lambda x, y: (x-gradient, y))
    model.force = lambda t: force.copy()
    # U=(x,-y), N=(x,y), Lap U=0, p=nu+gradient*(2-x), including outlet traction.
    np.testing.assert_allclose(model.nonlinear(velocity), model.project(lambda x, y: (x, y)), atol=4e-12)
    rate, pressure = model.instantaneous_solution(velocity, .3)
    np.testing.assert_allclose(rate, 0., atol=8e-11)
    np.testing.assert_allclose(pressure, nu+gradient*(2-model.pressure_basis.doflocs[0]), atol=8e-11)
    initial = model.state(0., velocity)
    for scheme in SCHEMES:
        trial = make_scheme(scheme).step(model, initial, .007)
        np.testing.assert_allclose(model.vector(trial.state), velocity, atol=8e-11)
        assert abs(trial.state.r) < 8e-11

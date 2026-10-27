"""Cached sampling, separable mass solves and common-matrix Stokes columns."""
import numpy as np
import pytest

from solver.spectral.assembly import Space
from solver.spectral.model import SpectralModel
from solver.spectral.mass import TensorMass
from solver.spectral.lifting import LidLifting
from solver.spectral.stokes import SpectralStokes
from solver.spectral.quadrature import QuadratureCache


def test_cached_rectangle_sampling_keeps_physical_coordinates_and_public_arrays(monkeypatch):
    space = Space(6, 1.3, .8)
    model = SpectralModel(space, .1)
    plan = model._quadrature.get(model.quadrature_extra)
    u = np.zeros((space.size, space.size))
    u[0, 0] = 1.
    x, y = np.meshgrid(plan.x, plan.y)
    # phi_0(x)=6*x/lx*(1-x/lx); this reference uses no basis recurrences.
    exact = 36*x/space.lx*(1-x/space.lx)*y/space.ly*(1-y/space.ly)
    np.testing.assert_allclose(plan.evaluate(u), exact, atol=2e-15)
    ux = 36/space.lx*(1-2*x/space.lx)*y/space.ly*(1-y/space.ly)
    np.testing.assert_allclose(plan.divergence(u, np.zeros_like(u)), ux, atol=1e-14)
    assert not plan.x.flags.writeable and not plan.values_x.flags.writeable
    public_x = space.nodes(extra=model.quadrature_extra)[0]
    public_x[:] = 0.
    assert np.all(plan.x > 0.)
    field = np.r_[u.ravel(), np.zeros(u.size)]
    expected = model.nonlinear(field)
    expected_diagnostic = model.diagnostics(model.state(0., field))

    def forbidden(*args, **kwargs):
        raise AssertionError('warm sampling rebuilt quadrature or bases')

    monkeypatch.setattr(Space, 'nodes', forbidden)
    monkeypatch.setattr(Space, 'velocity_values', forbidden)
    monkeypatch.setattr(Space, 'velocity_derivatives', forbidden)
    np.testing.assert_array_equal(model.nonlinear(field), expected)
    assert model.diagnostics(model.state(0., field)) == expected_diagnostic
    model.physical_divergence_inf(field)
    model.close()
    assert not model._quadrature.plans


def test_quadrature_cache_is_bounded_and_uses_recent_resolutions():
    cache = QuadratureCache(Space(4), capacity=2)
    first = cache.get(3)
    cache.get(4)
    assert cache.get(3) is first
    cache.get(5)
    assert list(cache.plans) == [3, 5]


@pytest.mark.parametrize('size', [1, 6, 32])
def test_tensor_mass_inverse_recovers_known_modal_coefficients_on_a_rectangle(size):
    from solver.spectral.assembly import velocity_mass
    space = Space(size, 1.3, .8)
    exact = np.random.default_rng(73).normal(size=size**2)
    mass = velocity_mass(space)
    recovered = TensorMass(space).solve(mass @ exact)
    np.testing.assert_allclose(recovered, exact, rtol=2e-12, atol=2e-12)
    assert np.linalg.norm(mass @ recovered-mass @ exact) < 1e-14


def test_projected_polynomial_keeps_coefficients_instead_of_weak_loads():
    space = Space(6, 1.3, .8)
    model = SpectralModel(space, .1)
    def exact(x, y):
        return x*(space.lx-x)*y*(space.ly-y)*(.7+2*x-y+x*y)
    coefficients = model.project(exact)
    x, y = np.meshgrid(np.linspace(0., space.lx, 17), np.linspace(0., space.ly, 13))
    values = space.evaluate(coefficients.reshape(6, 6), x[0], y[:, 0])
    np.testing.assert_allclose(values, exact(x, y), atol=2e-15)


def test_batched_stokes_recovers_independent_known_columns_and_checks_each_column():
    solver = SpectralStokes(Space(6, 1.3, .8), tolerance=1e-12)
    random = np.random.default_rng(92)
    velocities = [random.normal(size=(2, 36))*scale for scale in (1., 1e-4)]
    pressures = [random.normal(size=36)*scale for scale in (1., 1e-4)]
    for pressure in pressures:
        pressure[0] = 0.
    h = 1.7*solver.mass+.013*solver.stiffness
    loads_x = [h @ w[0]-solver.divergence_x.T @ p for w, p in zip(velocities, pressures)]
    loads_y = [h @ w[1]-solver.divergence_y.T @ p for w, p in zip(velocities, pressures)]
    constraints = [solver.divergence_x @ w[0]+solver.divergence_y @ w[1] for w in velocities]
    results = solver.solve_columns(loads_x, loads_y, mass=1.7, viscosity=.013, continuities=constraints)
    assert solver.factorizations == 1 and solver.solve_calls == 1 and solver.right_hand_sides == 2
    for solution, velocity, pressure in zip(results, velocities, pressures):
        np.testing.assert_allclose(solution.velocity_x.ravel(), velocity[0], atol=3e-14)
        np.testing.assert_allclose(solution.velocity_y.ravel(), velocity[1], atol=3e-14)
        np.testing.assert_allclose(solution.pressure.ravel(), pressure, atol=3e-14)
    lu, matrix = solver.cache[1.7, .013]
    class CorruptSecondColumn:
        def solve(self, load):
            result = lu.solve(load)
            result[0, 1] += 1e-3
            return result
    solver.cache[1.7, .013] = (CorruptSecondColumn(), matrix)
    with pytest.raises(RuntimeError, match='Spectral Stokes residual'):
        solver.solve_columns(loads_x, loads_y, mass=1.7, viscosity=.013, continuities=constraints)


def test_batched_model_keeps_lifting_only_on_first_column():
    space = Space(6, 1.3, .8)
    model = SpectralModel(space, .1, lifting=LidLifting(space))
    columns = [np.random.default_rng(j).normal(size=72)*.01 for j in (3, 4)]
    batched = model.solve_columns(columns, mass=1., viscosity=.001)
    singles = [model.solve(column, mass=1., viscosity=.001, with_boundary_data=j == 0)
               for j, column in enumerate(columns)]
    for first, second in zip(batched, singles):
        np.testing.assert_allclose(first.velocity, second.velocity, atol=3e-14)
        np.testing.assert_allclose(first.pressure, second.pressure, atol=3e-14)
    combined = model.combine((1., batched[0].velocity), (-.3, batched[1].velocity))
    assert model.max_abs(model.apply_D(combined)) < 1e-13


def test_factor_cache_reuses_exact_shifts_with_bounded_lru():
    solver = SpectralStokes(Space(4), cache_size=2)
    first = solver._system(1., .01)
    solver._system(1., .02)
    assert solver._system(1., .01) is first
    solver._system(1., .03)
    assert list(solver.cache) == [(1., .01), (1., .03)]

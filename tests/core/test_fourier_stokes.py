import numpy as np
import pytest

from solver.fourier import FourierSpace
from solver.fourier.stokes import FourierStokes


@pytest.mark.parametrize('mass,viscosity', [(1., .03), (0., .7), (2., 0.)])
def test_known_shifted_solution_pressure_full_equation_and_batch(mass, viscosity):
    s = FourierSpace(21, 18, 3., 4.)
    x, y = 2*np.pi*s.x/s.lx, 2*np.pi*s.y/s.ly
    exact = s.velocity((2*np.pi/s.ly)*np.sin(x)*np.cos(y),
                       -(2*np.pi/s.lx)*np.cos(x)*np.sin(y))
    p = s.to_spectral(.3*np.cos(2*x)*np.sin(y))
    rhs = mass*exact+viscosity*s.k2*exact+s.gradient(p)
    before = rhs.copy()
    backend = FourierStokes(s)
    results = backend.solve_columns([rhs, 2*rhs], mass=mass, viscosity=viscosity)
    np.testing.assert_array_equal(rhs, before)
    for factor, result in zip((1., 2.), results):
        np.testing.assert_allclose(result.velocity, factor*exact, atol=2e-14)
        np.testing.assert_allclose(result.pressure, factor*p, atol=2e-14)
        assert result.residual < 1e-14 and result.divergence_inf < 2e-14
        assert result.pressure[0, 0] == 0
    beta = s.inner(rhs, results[0].velocity)
    assert beta >= 0
    assert beta == pytest.approx(s.inner(mass*exact+viscosity*s.k2*exact, exact), abs=1e-12)


def test_gradient_and_mean_modes():
    s = FourierSpace(16, 13)
    backend = FourierStokes(s)
    p = s.to_spectral(np.cos(s.x)*np.sin(s.y))
    solved = backend.solve(s.gradient(p), mass=1., viscosity=.2)
    np.testing.assert_allclose(solved.velocity, 0, atol=1e-14)
    np.testing.assert_allclose(solved.pressure, p, atol=1e-14)
    rhs = s.velocity(np.full(s.shape, 2.), np.full(s.shape, -3.))
    solved = backend.solve(rhs, mass=2., viscosity=1.)
    np.testing.assert_allclose(s.to_physical(solved.velocity),
                               np.stack([np.ones(s.shape), np.full(s.shape, -1.5)]), atol=1e-14)
    with pytest.raises(ValueError, match='zero-mean'):
        backend.solve(rhs, mass=0., viscosity=1.)

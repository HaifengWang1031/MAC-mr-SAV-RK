import numpy as np
import pytest

from solver.fourier import FourierMHD, FourierSpace
from solver.integrate import integrate
from solver.schemes.registry import make_scheme


def test_projected_mhd_flux_and_joint_energy_pairing():
    s = FourierSpace(20, 19)
    model = FourierMHD(s, .02, .03)
    # Specify the analytic low-mode fields directly, not via the task's IC builder.
    fields = np.array([[-np.sin(s.y), np.sin(s.x)],
                       [-np.sin(s.y), np.sin(2*s.x)]])
    initial = model.state(0., s.to_spectral(fields))
    nonlinear = s.to_physical(model.nonlinear(model.vector(initial)))
    # Differentiate the exact low modes in (7) and remove the gradient analytically.
    x, y = s.x, s.y
    expected = np.array([
        [-.6*np.sin(2*x)*np.cos(y), 1.2*np.cos(2*x)*np.sin(y)],
        [(np.sin(2*x)-np.sin(x))*np.cos(y), (np.cos(x)-2*np.cos(2*x))*np.sin(y)],
    ])
    np.testing.assert_allclose(nonlinear, expected, atol=5e-14)
    np.testing.assert_allclose(
        [model.diagnostics(initial)[key] for key in ('kinetic', 'magnetic', 'dissipation')],
        [.5, .5, .02+2.5*.03], atol=1e-14)
    random = np.random.default_rng(223).normal(size=(2, 2, *s.shape))
    z = s.project(s.to_spectral(random))
    n = model.nonlinear(z)
    assert model.physical_divergence_inf(n) < 1e-12
    assert abs(model.inner(n, z)) < 1e-12
    with pytest.raises(RuntimeError, match='divergence'):
        model.solve(s.to_spectral(random), mass=1., viscosity=.01)


@pytest.mark.parametrize('name', ('sdirk3', 'sdirk3_mrsav'))
def test_exact_independently_diffusing_periodic_shears(name):
    s = FourierSpace(17, 16)
    model = FourierMHD(s, .2, .07)
    u, bx = np.sin(s.y), np.cos(2*s.y)
    zero = np.zeros(s.shape)
    initial = model.state(0., s.to_spectral(np.array([[u, zero], [bx, zero]])))
    result = integrate(model, make_scheme(name), initial, [.01]*16)
    assert result.status == 'complete', result.error
    t = result.final.t
    np.testing.assert_allclose(result.final.u, np.exp(-.2*t)*u, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.final.magnetic[0], np.exp(-4*.07*t)*bx, rtol=0, atol=1e-10)
    assert abs(result.final.r) < 1e-14
    assert all(stage.pressure.size == 0 for stage in result.final_stages)

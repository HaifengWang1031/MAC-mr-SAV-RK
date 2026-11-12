import numpy as np
import pytest

from solver.fourier import FourierSpace
from solver.fourier.forced_mhd import ForcedFourierMHD, flux_potential, transfer_spectra
from solver.integrate import integrate
from solver.schemes.registry import make_scheme


@pytest.mark.parametrize('scheme', ['sdirk3', 'sdirk3_mrsav'])
def test_forced_hyperdiffusive_shear_exact_solution(scheme):
    s = FourierSpace(18, 18)
    zero = np.zeros(s.shape)
    fields = np.array([[np.sin(2*s.y), zero], [np.cos(3*s.y), zero]])
    force = s.to_spectral(fields)
    model = ForcedFourierMHD(s, .01, .3, force=lambda t: force)
    initial = model.state(0., np.zeros_like(force))
    result = integrate(model, make_scheme(scheme), initial, [.005]*40)
    assert result.status == 'complete', result.error
    rates = np.array([.01*2**4+.3/2**4, .01*3**4+.3/3**4])
    expected = ((1-np.exp(-rates*.2))/rates)[:, None, None, None]*fields
    np.testing.assert_allclose(s.to_physical(model.vector(result.final)), expected, atol=2e-11)
    d = model.diagnostics(result.final)
    assert d['dissipation_E_small'] > 0 and d['dissipation_E_large'] > 0
    assert d['power_u'] > 0 and d['power_B'] > 0
    assert abs(result.final.r) < 1e-14
    np.testing.assert_allclose(d['dissipation'], np.sum(.5*rates*((1-np.exp(-rates*.2))/rates)**2), atol=1e-12)
    np.testing.assert_allclose(d['potential_squared'], .25*((1-np.exp(-rates[1]*.2))/rates[1])**2/9)


def test_flux_against_trigonometric_advective_reference():
    s = FourierSpace(30, 30)
    x, y = s.x, s.y
    theta, delta = x+2*y, x-y
    u, v = -2*np.sin(2*y)-2*np.sin(theta), np.sin(x)+np.sin(theta)
    bx = by = np.sin(delta)
    z = s.to_spectral(np.array([[u, v], [bx, by]]))
    model = ForcedFourierMHD(s, .01, .1, force=lambda t: np.zeros_like(z))
    # Analytic physical derivatives, independent of the conservative flux implementation.
    ux, uy = -2*np.cos(theta), -4*np.cos(2*y)-4*np.cos(theta)
    vx, vy = np.cos(x)+np.cos(theta), 2*np.cos(theta)
    bdx, bdy = np.cos(delta), -np.cos(delta)
    adv = np.array([[u*ux+v*uy-bx*bdx-by*bdy,
                     u*vx+v*vy-bx*bdx-by*bdy],
                    [u*bdx+v*bdy-bx*ux-by*uy,
                     u*bdx+v*bdy-bx*vx-by*vy]])
    spectra = transfer_spectra(model, z)
    a = np.cos(delta)
    np.testing.assert_allclose(s.to_physical(flux_potential(s, z[1])), a, atol=1e-14)
    for K in range(1, 5):
        low = s.to_physical(z*(s.k2 <= K*K))
        expected = np.mean(np.sum(low*adv, axis=(0, 1)))
        np.testing.assert_allclose(spectra['Pi_E'][K], expected, atol=5e-14)
        alow = s.to_physical(s.to_spectral(a)*(s.k2 <= K*K))
        np.testing.assert_allclose(spectra['Pi_Q'][K], np.mean(alow*(v-u)*np.sin(delta)), atol=5e-14)
    np.testing.assert_allclose(spectra['Pi_E'][1], .5, atol=1e-14)  # Low mode loses energy.
    reversed_flux = transfer_spectra(model, -z)
    np.testing.assert_allclose(reversed_flux['Pi_E'][1], -.5, atol=1e-14)
    assert abs(spectra['Pi_E'][-1]) < 1e-13
    assert abs(spectra['Pi_Q'][-1]) < 1e-13
    np.testing.assert_allclose(np.sum(spectra['E_u']+spectra['E_B']), .5*np.mean(u*u+v*v+bx*bx+by*by))


def test_damping_zero_mode_and_load_validation():
    s = FourierSpace(12, 12)
    zero = np.zeros((2, 2, *s.spectral_shape), complex)
    model = ForcedFourierMHD(s, .01, .1, force=lambda t: zero)
    z = zero.copy()
    z[..., 0, 0] = 1
    np.testing.assert_array_equal(model.solve(z, mass=1, viscosity=.1).velocity, z)
    with pytest.raises(ValueError):
        ForcedFourierMHD(s, .01, -1, force=lambda t: zero)
    bad = s.to_spectral(np.ones((2, 2, *s.shape))*np.sin(s.x))
    model = ForcedFourierMHD(s, .01, .1, force=lambda t: bad)
    with pytest.raises(ValueError, match='divergence'):
        model.force(0)

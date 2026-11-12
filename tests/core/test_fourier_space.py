import numpy as np
import pytest

from solver.fourier import FourierSpace


@pytest.mark.parametrize('nx,ny', [(18, 16), (19, 17)])
def test_rectangular_fft_parseval_derivatives_and_adjoint(nx, ny):
    s = FourierSpace(nx, ny, 3., 5.)
    x, y = 2*np.pi*s.x/s.lx, 2*np.pi*s.y/s.ly
    u = np.cos(2*x)*np.sin(y)+.2
    v = np.sin(x)*np.cos(2*y)-.3
    z = s.velocity(u, v, project=False)
    np.testing.assert_allclose(s.to_physical(z), [u, v], atol=1e-14)
    assert s.inner(z, z) == pytest.approx(s.area*np.mean(u*u+v*v), abs=1e-13)
    divergence = -4*np.pi/s.lx*np.sin(2*x)*np.sin(y)-4*np.pi/s.ly*np.sin(x)*np.sin(2*y)
    np.testing.assert_allclose(s.to_physical(s.divergence(z)), divergence, atol=2e-14)
    p = s.to_spectral(np.sin(x)*np.sin(y))
    assert s.inner(z, s.gradient(p)) == pytest.approx(-s.inner(s.divergence(z), p), abs=1e-13)
    projected = s.project(z)
    np.testing.assert_allclose(s.project(projected), projected, atol=1e-15)
    assert np.max(np.abs(s.divergence(projected))) < 1e-14
    np.testing.assert_allclose(projected[:, 0, 0], [.2, -.3], atol=1e-15)
    np.testing.assert_allclose(s.project(s.gradient(p)), 0, atol=1e-15)


def test_strict_cutoff_and_invalid_spectral_load():
    s = FourierSpace(18, 15)
    z = s.velocity(np.cos(6*s.x)+np.cos(5*s.x), np.zeros(s.shape), project=False)
    np.testing.assert_allclose(s.to_physical(z)[0], np.cos(5*s.x), atol=1e-14)
    assert not s.mask[:, 6].any() and not s.mask[5].any()
    assert s.metadata()['retained_full_modes'] == 11*9
    s.check_velocity(z)
    z[0, 0, 6] = 1.
    with pytest.raises(ValueError, match='outside'):
        s.check_velocity(z)

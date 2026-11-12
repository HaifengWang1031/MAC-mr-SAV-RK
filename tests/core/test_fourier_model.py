import numpy as np
import pytest

from solver.fourier import FourierModel, FourierSpace


def test_nonlinear_matches_analytic_product_is_unprojected_and_energy_skew():
    s = FourierSpace(24, 21)
    model = FourierModel(s, .1)
    x, y = s.x, s.y
    # psi=sin(x)sin(y)+.2 sin(2x)sin(y); this has a nonzero projected convection.
    u = (np.sin(x)+.2*np.sin(2*x))*np.cos(y)
    v = -(np.cos(x)+.4*np.cos(2*x))*np.sin(y)
    ux = (np.cos(x)+.4*np.cos(2*x))*np.cos(y)
    uy = -(np.sin(x)+.2*np.sin(2*x))*np.sin(y)
    vx = (np.sin(x)+.8*np.sin(2*x))*np.sin(y)
    vy = -(np.cos(x)+.4*np.cos(2*x))*np.cos(y)
    z = s.velocity(u, v)
    before = z.copy()
    convection = model.nonlinear(z)
    np.testing.assert_allclose(convection, s.velocity(u*ux+v*uy, u*vx+v*vy, project=False), atol=1e-14)
    np.testing.assert_array_equal(z, before)
    assert model.inner(z, convection) == pytest.approx(0., abs=2e-13)
    assert model.max_abs(s.project(convection)) > .001
    assert model.max_abs(s.divergence(convection)) > .001
    # Highest retained inputs: output matches a much wider-grid product, detecting
    # low-mode aliases that filtering only full-band product outputs would retain.
    wide = FourierSpace(72, 63)
    a, b = s.x, s.y
    high = s.velocity(np.sin(7*a)*np.cos(6*b), -7/6*np.cos(7*a)*np.sin(6*b))
    exact = FourierModel(wide, .1).nonlinear(wide.velocity(
        np.sin(7*wide.x)*np.cos(6*wide.y), -7/6*np.cos(7*wide.x)*np.sin(6*wide.y)))
    # Product only has modes 0 and +/-14 or +/-12; all nonzero outputs are cut here.
    np.testing.assert_allclose(model.nonlinear(high), 0, atol=2e-14)
    assert np.max(np.abs(exact)) > .01
    arbitrary = s.velocity(u+np.cos(3*x), v, project=False)
    assert abs(model.inner(arbitrary, model.nonlinear(arbitrary))) < 5e-13


def test_real_state_stage_and_physical_diagnostics():
    s = FourierSpace(19, 16)
    model = FourierModel(s, .1)
    z = s.velocity(np.sin(s.y)+.2, np.cos(s.x)-.1)
    state = model.state(0., z, .3)
    assert state.u.shape == state.v.shape == s.shape
    assert not np.iscomplexobj(state.u)
    np.testing.assert_allclose(model.vector(state), z, atol=1e-14)
    d = model.diagnostics(state)
    assert d['kinetic'] == pytest.approx(.5*s.area*np.mean(state.u**2+state.v**2))
    assert d['mean_u'] == pytest.approx(.2)
    assert d['mean_v'] == pytest.approx(-.1)
    assert d['enstrophy'] == pytest.approx(.5*d['h1_seminorm_squared'])
    p = s.to_spectral(np.sin(s.x)*np.cos(s.y))
    stage = model.stage(p, 0., 0.)
    np.testing.assert_allclose(stage.pressure, np.sin(s.x)*np.cos(s.y), atol=1e-14)

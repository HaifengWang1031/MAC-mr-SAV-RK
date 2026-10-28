import numpy as np
import pytest

from solver.channel import ChannelGrid, ChannelModel
from solver.channel.kernels import convection
from solver.mac.stokes import DirectStokes


def flux_reference(u, v, hx, hy):
    """Vectorised periodic fluxes independent of the compiled loop indices."""
    q = u[:, :-1]
    east, west = .5*(q+np.roll(q, -1, 1)), .5*(q+np.roll(q, 1, 1))
    north = .5*(v[1:]+np.roll(v[1:], 1, 1))
    south = .5*(v[:-1]+np.roll(v[:-1], 1, 1))
    padded = np.vstack([-q[:1], q, -q[-1:]])
    cu = (east*.5*(q+np.roll(q, -1, 1))-west*.5*(q+np.roll(q, 1, 1)))/hx
    cu += (north*.5*(q+padded[2:])-south*.5*(q+padded[:-2]))/hy
    cu -= .5*q*((east-west)/hx+(north-south)/hy)
    qv = v[1:-1]
    west = .5*(q[:-1]+q[1:]); east = np.roll(west, -1, 1)
    north, south = .5*(qv+v[2:]), .5*(qv+v[:-2])
    cv = (east*.5*(qv+np.roll(qv, -1, 1))-west*.5*(qv+np.roll(qv, 1, 1)))/hx
    cv += (north*.5*(qv+v[2:])-south*.5*(qv+v[:-2]))/hy
    cv -= .5*qv*((east-west)/hx+(north-south)/hy)
    return np.column_stack([cu, cu[:, 0]]), np.vstack([np.zeros_like(v[:1]), cv, np.zeros_like(v[:1])])


def test_periodic_flux_reference_skew_pairing_and_physical_divergence():
    model = ChannelModel(ChannelGrid(9, 7, 2., .8), .1)
    w = np.random.default_rng(43).normal(size=model.grid.size)
    u, v = model.grid.unpack(w)
    expected = flux_reference(u, v, model.grid.hx, model.grid.hy)
    actual = convection(u, v, model.grid.hx, model.grid.hy)
    for a, b in zip(actual, expected): np.testing.assert_allclose(a, b, atol=1e-13)
    n = model.nonlinear(w)
    assert abs(model.inner(n, w)) < 2e-13*max(1., model.grid.norm(n)*model.grid.norm(w))
    assert model.physical_divergence_inf(w) == pytest.approx(model.max_abs(model.apply_D(w)))


def test_gradient_only_pressure_batch_cache_and_nonnegative_sav_beta():
    model = ChannelModel(ChannelGrid(7, 5, 1.3, .8), .13, cache_size=2)
    rng = np.random.default_rng(44)
    p = rng.normal(size=(5, 7)); p -= p.mean()
    b = model.apply_G(p)
    random_load = rng.normal(size=model.grid.size)
    solves = model.solve_columns([b, random_load], mass=1., viscosity=.02)
    np.testing.assert_allclose(solves[0].velocity, 0., atol=2e-13)
    np.testing.assert_allclose(solves[0].pressure, p, atol=2e-13)
    assert model.inner(random_load, solves[1].velocity) >= -1e-13
    for sol, rhs in zip(solves, [b, random_load]):
        np.testing.assert_allclose(sol.velocity+.02*model.apply_K(sol.velocity)
                                   +model.apply_G(sol.pressure), rhs, atol=2e-13)
        assert model.physical_divergence_inf(sol.velocity) < 1e-12
        assert abs(sol.pressure.mean()) < 1e-14
    assert isinstance(model.backend, DirectStokes)
    model.solve(b, mass=1., viscosity=.02)
    assert model.backend.factorizations == 1
    for alpha in [.03, .04]: model.solve(b, mass=1., viscosity=alpha)
    assert len(model.backend.cache) == 2


def test_mean_driving_force_discrete_poiseuille_and_second_order():
    errors = []
    for ny in [8, 16, 32]:
        grid = ChannelGrid(5, ny, 2., .7)
        force = grid.pack_load(np.ones((ny, 6)), np.zeros((ny+1, 5)))
        model = ChannelModel(grid, .13, force=lambda t: force)
        sol = model.solve(force, mass=0., viscosity=model.nu)
        _, y = grid.coordinates('u')
        exact = y*(grid.ly-y)/(2*model.nu)
        discrete = exact+grid.hy**2/(8*model.nu)
        u, v = grid.unpack(sol.velocity)
        np.testing.assert_allclose(u, discrete, atol=2e-12)
        np.testing.assert_allclose(v, 0., atol=2e-12)
        errors.append(grid.norm(sol.velocity-grid.pack(exact, v*0)))
        assert model.max_abs(model.nonlinear(sol.velocity)) < 1e-12
        diagnostics = model.diagnostics(model.state(0., sol.velocity))
        assert diagnostics['flow_rate_spread'] < 1e-12
        assert diagnostics['power'] == pytest.approx(diagnostics['dissipation'], abs=1e-12)
    np.testing.assert_allclose(np.array(errors[:-1])/errors[1:], 4., rtol=1e-10)


def test_wall_normal_constant_load_is_balanced_by_pressure():
    grid = ChannelGrid(6, 7, 1.2, .9)
    model = ChannelModel(grid, .1)
    load = grid.pack_load(np.zeros((7, 7)), np.ones((8, 6)))
    sol = model.solve(load, mass=0., viscosity=.1)
    _, y = grid.coordinates('p')
    np.testing.assert_allclose(sol.velocity, 0., atol=1e-13)
    np.testing.assert_allclose(sol.pressure, y-grid.ly/2, atol=1e-13)

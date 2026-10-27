import numpy as np
import pytest

from experiments.channel_ns.problems import DiscreteManufactured, SCHEMES, scheme
from solver.adaptivity import PIController, integrate_adaptive
from solver.channel import ChannelGrid, ChannelModel, FFTStokes
from solver.integrate import integrate


@pytest.mark.parametrize('nx,ny', [(7, 5), (8, 6), (17, 9)])
@pytest.mark.parametrize('mass,alpha', [(1., 0.), (1., .013), (0., .2)])
def test_fft_banded_matches_global_direct_with_full_equations(nx, ny, mass, alpha):
    grid = ChannelGrid(nx, ny, 2.3, .8)
    direct, fft = ChannelModel(grid, .1), ChannelModel(grid, .1, backend='fft')
    loads = np.random.default_rng(54).normal(size=(grid.size, 2))
    a, b = direct.solve(loads, mass=mass, viscosity=alpha), fft.solve(loads, mass=mass, viscosity=alpha)
    np.testing.assert_allclose(a.velocity, b.velocity, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(a.pressure, b.pressure, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(mass*b.velocity+alpha*fft.apply_K(b.velocity)
                               +fft.ops.G@b.pressure.reshape(grid.np, 2), loads, atol=2e-12)
    assert b.divergence_inf < 1e-11


def test_zero_mode_mean_drive_nonzero_mode_pressure_means_and_cache():
    grid = ChannelGrid(8, 7, 1.2, .9)
    model = ChannelModel(grid, .1, backend='fft', cache_size=2)
    x, y = grid.coordinates('p')
    p = np.cos(2*np.pi*x/grid.lx)+y-grid.ly/2
    sol = model.solve(model.apply_G(p), mass=1., viscosity=.1)
    np.testing.assert_allclose(sol.velocity, 0., atol=1e-13)
    np.testing.assert_allclose(sol.pressure, p, atol=1e-13)
    force = grid.pack_load(np.ones((7, 9)), np.zeros((8, 8)))
    sol = model.solve(force, mass=0., viscosity=.1)
    _, yu = grid.coordinates('u')
    expected = (yu*(grid.ly-yu)+grid.hy**2/4)/.2
    np.testing.assert_allclose(grid.unpack(sol.velocity)[0], expected, atol=1e-12)
    assert isinstance(model.backend, FFTStokes)
    model.solve(force, mass=0., viscosity=.1)
    assert model.backend.factorizations == 2
    model.solve(force, mass=1., viscosity=.2)
    assert len(model.backend.cache) == 2


@pytest.mark.parametrize('name', SCHEMES)
def test_identical_time_schemes_across_channel_backends(name):
    grid = ChannelGrid(12, 9, 2*np.pi, 1.)
    reference = DiscreteManufactured(grid, .1)
    models = [ChannelModel(grid, .1, force=reference.force, backend=b) for b in ('direct', 'fft')]
    results = [integrate(m, scheme(name, 1000.), m.state(0., reference.exact(0.)), [.01, .02, .01]) for m in models]
    assert all(r.status == 'complete' for r in results), [r.error for r in results]
    for component in ('u', 'v'):
        np.testing.assert_allclose(getattr(results[0].final, component), getattr(results[1].final, component), atol=1e-12)
    assert abs(results[0].final.r-results[1].final.r) < 1e-12
    for a, b in zip(results[0].final_stages, results[1].final_stages):
        np.testing.assert_allclose(a.pressure, b.pressure, atol=1e-11)


def test_fft_adaptive_pi_endpoint_and_field_agreement():
    grid = ChannelGrid(10, 8, 2*np.pi, 1.)
    reference = DiscreteManufactured(grid, .1)
    results = []
    for backend in ('direct', 'fft'):
        model = ChannelModel(grid, .1, force=reference.force, backend=backend)
        results.append(integrate_adaptive(model, scheme('sdirk3_mrsav'), model.state(0., reference.exact(0.)),
                                          .1, .1, controller=PIController(atol=1e-9, rtol=1e-6)))
    assert all(r.status == 'complete' and r.final.t == .1 for r in results)
    np.testing.assert_allclose(results[0].times, results[1].times, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(results[0].final.u, results[1].final.u, atol=1e-9)

import numpy as np
import pytest

from solver.channel import ChannelGrid, ChannelOperators


def test_layout_roundtrip_unique_periodic_weights_and_loads():
    grid = ChannelGrid(7, 5, 2., .7)
    w = np.random.default_rng(13).normal(size=grid.size)
    u, v = grid.unpack(w)
    np.testing.assert_array_equal(u[:, 0], u[:, -1])
    np.testing.assert_array_equal(v[[0, -1]], 0.)
    np.testing.assert_array_equal(grid.pack(u, v), w)
    expected = grid.area*(np.sum(u[:, :-1]**2)+np.sum(v[1:-1]**2))
    assert grid.inner(w, w) == pytest.approx(expected)
    v[[0, -1]] = 2.
    np.testing.assert_array_equal(grid.pack_load(u, v), w)
    with pytest.raises(ValueError, match='wall'): grid.pack(u, v)
    u[:, -1] += 1.
    with pytest.raises(ValueError, match='periodic'): grid.pack_load(u, v)
    with pytest.raises(ValueError): grid.unpack(w.astype(complex))
    u[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'): grid.pack_load(u, v)


@pytest.mark.parametrize('nx,ny', [(7, 5), (8, 6)])
def test_laplacian_analytic_discrete_modes_including_wall_rows(nx, ny):
    grid = ChannelGrid(nx, ny, 2.3, .8)
    ops = ChannelOperators(grid)
    xu, yu = grid.coordinates('u')
    xv, yv = grid.coordinates('v')
    u = np.cos(4*np.pi*xu/grid.lx)*np.sin(np.pi*yu/grid.ly)
    u[:, -1] = u[:, 0]
    v = np.cos(4*np.pi*xv/grid.lx)*np.sin(np.pi*yv/grid.ly)
    v[[0, -1]] = 0.
    w = grid.pack(u, v)
    eigenvalue = 4*np.sin(2*np.pi/nx)**2/grid.hx**2 \
        + 4*np.sin(np.pi/(2*ny))**2/grid.hy**2
    np.testing.assert_allclose(ops.K@w, eigenvalue*w, rtol=2e-13, atol=2e-13)
    assert np.linalg.eigvalsh(ops.K.toarray())[0] > 0.


def test_adjoint_pressure_kernel_full_divergence_and_curl():
    grid = ChannelGrid(7, 5, 1.3, .8)
    ops = ChannelOperators(grid)
    rng = np.random.default_rng(31)
    w, p = rng.normal(size=grid.size), rng.normal(size=grid.np)
    assert np.dot(w, ops.G@p) == pytest.approx(-np.dot(ops.D@w, p), abs=1e-12)
    np.testing.assert_array_equal(ops.G@np.ones(grid.np), 0.)
    assert np.linalg.matrix_rank(ops.D.toarray()) == grid.np-1
    u, v = grid.unpack(w)
    expected = np.diff(u, axis=1)/grid.hx+np.diff(v, axis=0)/grid.hy
    np.testing.assert_allclose(ops.D@w, expected.ravel(), atol=1e-13)
    x, y = grid.coordinates('vertex')
    psi = np.sin(2*np.pi*x/grid.lx)*np.sin(np.pi*y/grid.ly)**2
    psi[[0, -1]] = 0.
    np.testing.assert_allclose(ops.D@grid.curl(psi), 0., atol=1e-13)


@pytest.mark.parametrize('args', [(True, 5), (2, 5), (5, 4, 0.), (5, 4, 1., np.inf)])
def test_invalid_grid(args):
    with pytest.raises(ValueError): ChannelGrid(*args)

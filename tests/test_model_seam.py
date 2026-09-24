"""Seam-member tests: every adapted member against an independent reference.

Seams confirmed in docs/model-seam-spec.md (member level). The references here are
deliberately independent of the implementation — the analytic discrete eigenvalue of
the sine mode for K, the transpose identity for G/D, and the discrete curl for D —
so none of them can pass by recomputing what the code computes.
"""
import numpy as np
from solver.mac.grid import MACGrid
from solver.mac_ns import MACNavierStokes


def seam_model() -> MACNavierStokes:
    return MACNavierStokes(MACGrid(9, 7, 1.3, .8), .1)


def sine_mode(grid: MACGrid) -> tuple[np.ndarray, np.ndarray]:
    """sin(pi x/lx) sin(pi y/ly) sampled on u and v faces, walls zeroed.

    The float value of sin(pi) is not exactly zero, so the eliminated normal boundary
    faces are cleared explicitly; that is the same odd extension the operator assumes.
    """
    hx, hy, lx, ly = grid.hx, grid.hy, grid.lx, grid.ly
    xu, yu = np.arange(grid.nx + 1) * hx, (np.arange(grid.ny) + .5) * hy
    xv, yv = (np.arange(grid.nx) + .5) * hx, np.arange(grid.ny + 1) * hy
    u = np.sin(np.pi * np.outer(np.ones_like(yu), xu) / lx) * np.sin(np.pi * np.outer(yu, np.ones_like(xu)) / ly)
    v = np.sin(np.pi * np.outer(np.ones_like(yv), xv) / lx) * np.sin(np.pi * np.outer(yv, np.ones_like(xv)) / ly)
    u[:, [0, -1]] = 0.
    v[[0, -1], :] = 0.
    return u, v


def test_apply_K_matches_the_analytic_discrete_eigenvalue():
    model = seam_model()
    grid = model.grid
    u, v = sine_mode(grid)
    field = grid.pack(u, v)
    # Exact discrete eigenvalue of the sine mode for this 5-point operator, derived from
    # the recurrence rather than from the assembled matrix. It holds on every row,
    # including the wall-adjacent ones where the diagonal is 3/h**2.
    expected = ((2 - 2 * np.cos(np.pi * grid.hx / grid.lx)) / grid.hx ** 2
                + (2 - 2 * np.cos(np.pi * grid.hy / grid.ly)) / grid.hy ** 2)
    np.testing.assert_allclose(model.apply_K(field), expected * field, rtol=1e-12, atol=1e-12)


def test_apply_G_is_the_negative_transpose_of_apply_D_and_annihilates_constants():
    model = seam_model()
    grid = model.grid
    rng = np.random.default_rng(1)
    velocity = rng.standard_normal(grid.size)
    pressure = rng.standard_normal(grid.np)
    assert model.apply_G(pressure).shape == (grid.size,)
    assert model.apply_D(velocity).shape == (grid.np,)
    # <Gp, v> = -<p, Dv> with unweighted sums: this is the transpose identity of the
    # compatible pair, independent of how either operator is stored.
    np.testing.assert_allclose(float(np.dot(model.apply_G(pressure), velocity)),
                               -float(np.dot(pressure, model.apply_D(velocity))), rtol=1e-12)
    # A constant pressure carries no gradient: D.T @ 1 vanishes exactly for this pair.
    np.testing.assert_array_equal(model.apply_G(np.ones(grid.np)), np.zeros(grid.size))


def test_apply_D_annihilates_the_discrete_curl():
    model = seam_model()
    grid = model.grid
    # u = d psi/dy, v = -d psi/dx telescopes to exactly zero divergence.
    psi = np.random.default_rng(2).standard_normal((grid.ny + 1, grid.nx + 1))
    psi[[0, -1], :] = 0.
    psi[:, [0, -1]] = 0.
    u = np.diff(psi, axis=0) / grid.hy
    v = -np.diff(psi, axis=1) / grid.hx
    np.testing.assert_allclose(model.apply_D(grid.pack(u, v)), 0., atol=1e-12)



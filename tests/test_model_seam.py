"""Seam-member tests: every adapted member against an independent reference.

Seams confirmed in docs/model-seam-spec.md (member level). The references here are
deliberately independent of the implementation — exact rational arithmetic for
combine, the analytic discrete eigenvalue of the sine mode for K, the transpose
identity for G/D, and a separate summation for inner — so none of them can pass by
recomputing what the code computes.
"""
import math
from fractions import Fraction
import numpy as np
from solver.mac.grid import MACGrid
from solver.mac_ns import MACNavierStokes
from solver.model import Model


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


def test_serial_model_satisfies_the_protocol():
    """Structural conformance, checked by the type checker rather than asserted at runtime."""
    model: Model = seam_model()
    assert model.nu == .1


def test_combine_is_exact_and_leaves_its_inputs_alone():
    model = seam_model()
    n = model.grid.size
    coefficients = (.5, -.25, 2.)
    vectors = (np.arange(1, n + 1, dtype=float),
               (np.arange(n, dtype=float) % 7) - 3.,
               np.ones(n))
    snapshots = [v.copy() for v in vectors]
    out = model.combine((coefficients[0], vectors[0]), (coefficients[1], vectors[1]),
                        (coefficients[2], vectors[2]))
    # Exact reference: every value and intermediate is a dyadic rational, so the result
    # can be compared to rational arithmetic bit for bit, in any association order.
    for k in range(n):
        exact = (Fraction(1, 2) * Fraction(vectors[0][k]) + Fraction(-1, 4) * Fraction(vectors[1][k])
                 + 2 * Fraction(vectors[2][k]))
        assert Fraction(float(out[k])) == exact, k
    for original, snapshot in zip(vectors, snapshots):
        assert np.array_equal(original, snapshot), 'combine must not mutate its inputs'
    assert np.array_equal(model.combine((1., vectors[0]), (-1., vectors[0])), np.zeros(n))


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


def test_inner_weights_by_cell_area_and_max_abs_matches_an_explicit_maximum():
    model = seam_model()
    grid = model.grid
    rng = np.random.default_rng(3)
    left = rng.standard_normal(grid.size)
    right = rng.standard_normal(grid.size)
    u, v = grid.unpack(left)
    ub, vb = grid.unpack(right)
    # Independent evaluation: explicit face loops, summed with math.fsum, so a missing or
    # wrong cell-area weight cannot cancel out.
    expected = grid.area * (math.fsum(float(u[j, i] * ub[j, i]) for j in range(grid.ny) for i in range(1, grid.nx))
                            + math.fsum(float(v[j, i] * vb[j, i]) for j in range(1, grid.ny) for i in range(grid.nx)))
    np.testing.assert_allclose(model.inner(left, right), expected, rtol=1e-13)
    np.testing.assert_allclose(model.max_abs(left), max(float(np.max(left)), -float(np.min(left))))
    exact = np.zeros(grid.size)
    exact[7] = 3.
    exact[3] = -1.
    assert model.max_abs(exact) == 3.

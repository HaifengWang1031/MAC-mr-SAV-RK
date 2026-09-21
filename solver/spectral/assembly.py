"""Tensor-product assembly for the single-element Legendre space.

A scalar field is represented in the velocity basis (Dirichlet-combined Legendre in
both directions) or in the pressure basis (plain Legendre truncated to the same index
range), with coefficients laid out as C[k, l] and flattened row-major, which is exactly
the ordering `scipy.sparse.kron` assumes. The 2D operators are therefore Kronecker sums
of the exact 1D matrices in `basis.py` and stay sparse.

Scaling follows x = (lx/2)(xi+1): a mass gains length/2, a derivative 2/length, and a
first derivative *inside* an integral gains neither, because the two cancel. That is
why the divergence blocks below carry no per-direction factor of their own.
"""
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from . import basis

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Space:
    """A single-element spectral space on [0, lx] x [0, ly] with `size` modes per direction."""

    size: int
    lx: float = 1.
    ly: float = 1.

    @property
    def velocity_dofs(self) -> int:
        return self.size * self.size

    def nodes(self, extra: int = 3) -> tuple[Array, Array, Array, Array]:
        """Gauss-Legendre nodes and weights per direction, in physical coordinates.

        Both directions are returned because they differ as soon as lx != ly; an earlier
        version scaled the y nodes by lx, which pushed the reference coordinate outside
        [-1, 1] and silently destroyed every quadrature.
        """
        reference, weights = leggauss(self.size + extra)
        return ((self.lx / 2) * (reference + 1), (self.lx / 2) * weights,
                (self.ly / 2) * (reference + 1), (self.ly / 2) * weights)

    def velocity_values(self, x: Array, y: Array) -> tuple[Array, Array]:
        """phi_k(x) and psi_l(y) for all modes, shapes (size, len(x)) and (size, len(y))."""
        reference_x = 2 * np.asarray(x) / self.lx - 1
        reference_y = 2 * np.asarray(y) / self.ly - 1
        return (basis.dirichlet_values(self.size, reference_x),
                basis.dirichlet_values(self.size, reference_y))

    def velocity_derivatives(self, x: Array, y: Array) -> tuple[Array, Array]:
        """d/dx phi_k(x) and d/dy psi_l(y), carrying the physical chain-rule factor."""
        reference_x = 2 * np.asarray(x) / self.lx - 1
        reference_y = 2 * np.asarray(y) / self.ly - 1
        return (basis.dirichlet_derivatives(self.size, reference_x) * (2 / self.lx),
                basis.dirichlet_derivatives(self.size, reference_y) * (2 / self.ly))

    def pressure_values(self, x: Array, y: Array) -> tuple[Array, Array]:
        """chi_m(x) and eta_n(y), the plain Legendre basis the pressure space uses.

        The pressure basis is not the Dirichlet-combined one, so evaluating a pressure with
        `evaluate` silently returns a different field; that mistake made the manufactured
        pressure look non-convergent while the velocity was already at machine precision.
        """
        reference_x = 2 * np.asarray(x) / self.lx - 1
        reference_y = 2 * np.asarray(y) / self.ly - 1
        return (basis.legendre_values(self.size, reference_x)[:self.size],
                basis.legendre_values(self.size, reference_y)[:self.size])

    def evaluate_pressure(self, coefficients: Array, x: Array, y: Array) -> Array:
        """Values of sum C[m,n] chi_m(x) eta_n(y), shaped like meshgrid(x, y) i.e. [y, x]."""
        chi, eta = self.pressure_values(x, y)
        return eta.T @ coefficients.T @ chi

    def project(self, values: Array, extra: int = 3) -> Array:
        """Galerkin load int f phi_k psi_l for `values` sampled on nodes(extra).

        `values` must be indexed [y, x] as meshgrid produces it. Exposed separately from
        `load` because the nonlinearity is assembled on its own, wider quadrature: with
        `extra` large enough the projection of a polynomial product is exact, which is what
        lets the convective term be checked against an analytic value.
        """
        nodes_x, weights_x, nodes_y, weights_y = self.nodes(extra=extra)
        phi, psi = self.velocity_values(nodes_x, nodes_y)
        weighted_x = phi * weights_x[None, :]
        weighted_y = psi * weights_y[None, :]
        # meshgrid varies x along columns, so `values` is indexed [y, x] and has to be
        # transposed before contacting the mode index; the result is [x, y] like every
        # other coefficient array here. The order cannot be settled by reasoning about
        # index conventions: on a square domain both orders agree, and an earlier
        # "discriminator" was ambiguous because its non-zero pattern is symmetric. It is
        # settled by comparing against sympy on a rectangle, where the un-transposed form
        # swaps the (0,2) and (2,0) entries, and by the separable test in the test file.
        return weighted_x @ values.T @ weighted_y.T

    def load(self, function: Callable[[Array, Array], Array], extra: int = 3) -> Array:
        """Galerkin load int f phi_k psi_l dx dy by Gauss-Legendre quadrature.

        Exact when the integrand is a polynomial of the degree the nodes support, and
        spectrally accurate for the smooth manufactured forcings used in the tests.
        """
        nodes_x, _, nodes_y, _ = self.nodes(extra=extra)
        grid_x, grid_y = np.meshgrid(nodes_x, nodes_y)
        return self.project(function(grid_x, grid_y), extra=extra)

    def evaluate(self, coefficients: Array, x: Array, y: Array) -> Array:
        """Values of sum C[k,l] phi_k(x) psi_l(y), shaped like meshgrid(x, y) i.e. [y, x].

        The coefficient matrix is indexed [x mode, y mode], so it has to be transposed
        before contacting the mode axis of psi; without that this returns the field of the
        transposed coefficients, which is invisible whenever those coefficients are
        symmetric (a square domain, or a field built from phi_0 psi_0) and wrong otherwise.
        """
        phi, psi = self.velocity_values(x, y)
        return psi.T @ coefficients.T @ phi

    def l2_norm(self, coefficients: Array) -> float:
        """||f||_2 of the represented field, from the exact velocity mass matrix."""
        mass = velocity_mass(self)
        flat = np.asarray(coefficients).reshape(-1)
        return float(np.sqrt(abs(flat @ (mass @ flat))))

    def _sampling(self, values: Array) -> tuple[Array, Array, Array, Array]:
        """Quadrature nodes matching a sampled [y, x] array, whose size fixes `extra`.

        The caller samples on `nodes(extra)`, so the node count is read back from the array
        instead of passed twice: a mismatch between the two is silent in the loads and only
        shows up as a wrong answer.
        """
        sampled = np.asarray(values)
        if sampled.ndim != 2 or sampled.shape[0] != sampled.shape[1] or sampled.shape[0] <= self.size:
            raise ValueError('Samples must be square [y, x] on nodes(extra) with extra > 0')
        return self.nodes(extra=sampled.shape[0] - self.size)

    def stiffness_load(self, derivative_x: Array, derivative_y: Array) -> Array:
        """Weak loads int (a d_x + b d_y)(phi_k psi_l) for a sampled gradient (a, b).

        These are the momentum loads of a known field g, which is how a nonzero Dirichlet
        value enters: the trial basis vanishes on the walls, so the wall value is split off
        as a known g and its loads move to the right-hand side. For g that lies in the
        space this returns exactly `velocity_stiffness @ coefficients`, which is the
        identity the test pins down.
        """
        nodes_x, weights_x, nodes_y, weights_y = self._sampling(derivative_x)
        phi, psi = self.velocity_values(nodes_x, nodes_y)
        phi_prime, psi_prime = self.velocity_derivatives(nodes_x, nodes_y)
        weight = weights_y[:, None] * weights_x[None, :]
        first = psi @ (np.asarray(derivative_x) * weight) @ phi_prime.T
        second = psi_prime @ (np.asarray(derivative_y) * weight) @ phi.T
        return first.T + second.T

    def divergence_load(self, divergence: Array) -> Array:
        """Weak loads int d chi_m eta_n of the divergence of a known field.

        Indexed [m, n] like the pressure coefficients, including the constant mode, which
        the Stokes solve drops as the gauge. The quadrature convention is pinned by the
        same identity as `stiffness_load`: for a field in the velocity space this must
        reproduce the assembled divergence blocks.
        """
        nodes_x, weights_x, nodes_y, weights_y = self._sampling(divergence)
        chi, eta = self.pressure_values(nodes_x, nodes_y)
        weight = weights_y[:, None] * weights_x[None, :]
        return (eta @ (np.asarray(divergence) * weight) @ chi.T).T


def _sparse(matrix: Array) -> sp.csr_matrix:
    """One sparse factor for kron: passing a dense array would switch its output type."""
    return sp.csr_matrix(np.atleast_2d(matrix))


def velocity_mass(space: Space) -> sp.csr_matrix:
    """int phi_k psi_l phi_m psi_n dx dy, block diagonal in the two velocity components."""
    one = basis.mass_reference(space.size) * basis.mass_scale(space.lx)
    two = basis.mass_reference(space.size) * basis.mass_scale(space.ly)
    return sp.kron(_sparse(one), _sparse(two), format='csr')


def velocity_stiffness(space: Space) -> sp.csr_matrix:
    """int grad u : grad v for one velocity component."""
    one = basis.stiffness_reference(space.size) * basis.stiffness_scale(space.lx)
    two = basis.mass_reference(space.size) * basis.mass_scale(space.ly)
    swapped = basis.mass_reference(space.size) * basis.mass_scale(space.lx)
    other = basis.stiffness_reference(space.size) * basis.stiffness_scale(space.ly)
    return (sp.kron(_sparse(one), _sparse(two), format='csr')
            + sp.kron(_sparse(swapped), _sparse(other), format='csr')).tocsr()


def divergence_blocks(space: Space) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """D_x and D_y mapping velocity coefficients to pressure coefficients.

    Test functions are chi_m(x) eta_n(y) with both factors plain Legendre, while trial
    functions are phi_k(x) psi_l(y) with both factors Dirichlet-combined. A derivative in
    one direction therefore pairs a divergence matrix with a pressure-times-velocity mass
    in the other, and no per-direction scale appears because the two factors cancel.

    D_x takes the x derivative, so its y factor is int eta_n psi_l dy, i.e. the projection
    matrix. D_y takes the y derivative, so its x factor is int chi_m phi_k dx instead. An
    earlier version used the velocity mass in D_x's y factor, which silently enforced a
    different constraint and kept the manufactured-solution error independent of the
    resolution.
    """
    projection_y = basis.projection_reference(space.size) * basis.mass_scale(space.ly)
    projection_x = basis.projection_reference(space.size) * basis.mass_scale(space.lx)
    return (sp.kron(_sparse(basis.divergence_reference(space.size)), _sparse(projection_y), format='csr'),
            sp.kron(_sparse(projection_x), _sparse(basis.divergence_reference(space.size)), format='csr'))

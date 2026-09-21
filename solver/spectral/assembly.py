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

    def load(self, function: Callable[[Array, Array], Array]) -> Array:
        """Galerkin load int f phi_k psi_l dx dy by Gauss-Legendre quadrature.

        Exact when the integrand is a polynomial of the degree the nodes support, and
        spectrally accurate for the smooth manufactured forcings used in the tests.
        """
        nodes_x, weights_x, nodes_y, weights_y = self.nodes()
        grid_x, grid_y = np.meshgrid(nodes_x, nodes_y)
        values = function(grid_x, grid_y)
        phi, psi = self.velocity_values(nodes_x, nodes_y)
        weighted_x = phi * weights_x[None, :]
        weighted_y = psi * weights_y[None, :]
        # `values` is indexed [y, x] because meshgrid(nodes_x, nodes_y) puts x in the
        # columns, so the y weights contract its rows and the x weights its columns, and
        # the result is [x, y] like every other coefficient array here. The decisive check
        # is analytic: int x phi_k psi_l is non-zero only in column zero, and int y phi_k
        # psi_l only in row zero. Two earlier attempts to "fix" this line transposed it the
        # wrong way; the discriminator above is what settles it.
        return weighted_y @ values @ weighted_x.T

    def evaluate(self, coefficients: Array, x: Array, y: Array) -> Array:
        """Values of sum C[k,l] phi_k(x) psi_l(y), shaped like meshgrid(x, y) i.e. [y, x]."""
        phi, psi = self.velocity_values(x, y)
        return psi.T @ coefficients @ phi

    def l2_norm(self, coefficients: Array) -> float:
        """||f||_2 of the represented field, from the exact velocity mass matrix."""
        mass = velocity_mass(self)
        flat = np.asarray(coefficients).reshape(-1)
        return float(np.sqrt(abs(flat @ (mass @ flat))))


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

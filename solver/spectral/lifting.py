"""Inhomogeneous Dirichlet values, split off as a known field.

The velocity basis vanishes on all four walls, so a nonzero wall value cannot be carried
by the unknowns. Following shenfun's construction, the wall profile is carried by two
extra basis functions, `(1+y/ly)/2` and `(1-y/ly)/2`, whose coefficients are *known*; their
contribution is therefore moved to the right-hand side. Here that contribution is written
as an explicit field g and the loads come from `Space.stiffness_load` and
`Space.divergence_load`, which leaves the saddle-point matrix untouched -- the same move
the MAC cavity makes when it puts the lid into a ghost-point viscous load.

The lid profile is first projected onto the 1D Dirichlet basis, exactly as shenfun's
`set_boundary_dofs` does. For the regularised lid `(1-xi^2)^2` that projection is exact and
the lifting is smooth; for the constant lid it is not, so the sharp lid arrives with a
rounded-off corner and Gibbs oscillations, which is what limits spectral accuracy there and
why the two variants are run side by side.
"""
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from . import basis
from .assembly import Space

Array = NDArray[np.float64]


def regularised(ξ: Array) -> Array:
    """The tutorial's corner-free lid, (1-xi^2)^2, in the reference coordinate."""
    return (1 - np.asarray(ξ) ** 2) ** 2


@dataclass
class LidLifting:
    """`g_u = U(x) y/ly` with `U` the 1D Dirichlet projection of the lid, `g_v = 0`.

    The vertical factor is shenfun's `(1+y/ly)/2`, whose coefficient at the top is the lid
    value and at the bottom zero, so this lifting imposes a moving top lid and stationary
    bottom over a domain [0, lx] x [0, ly].
    """

    space: Space
    profile: Callable[[Array], Array] = lambda ξ: np.ones_like(np.asarray(ξ))
    extra: int = 4

    def __post_init__(self) -> None:
        self.coefficients = self._project()

    def _project(self) -> Array:
        """Coefficients of the lid profile in the 1D Dirichlet basis.

        Exact for a profile the basis can represent, spectrally accurate otherwise; the
        quadrature is wide enough that the projection of the lid profiles used here carries
        no error of its own.
        """
        nodes, weights = leggauss(self.space.size + self.extra)
        values = basis.dirichlet_values(self.space.size, nodes)
        load = (weights[None, :] * values) @ np.asarray(self.profile(nodes))
        return np.linalg.solve(basis.mass_reference(self.space.size), load)

    def _profile_and_derivative(self, x: Array) -> tuple[Array, Array]:
        ξ = 2 * np.asarray(x) / self.space.lx - 1
        values = basis.dirichlet_values(self.space.size, ξ)
        derivative = basis.dirichlet_derivatives(self.space.size, ξ) * (2 / self.space.lx)
        return values.T @ self.coefficients, derivative.T @ self.coefficients

    def samples(self, x: Array, y: Array) -> tuple[Array, Array, Array, Array]:
        """(g_u, d_x g_u, d_y g_u, div g) sampled on the meshgrid of x and y, [y, x] shaped."""
        profile, profile_derivative = self._profile_and_derivative(x)
        # shenfun's (1+y/ly)/2, which is 0 at the bottom wall and 1 at the top.
        ramp = np.asarray(y) / self.space.ly
        slope = np.full(np.shape(ramp), 1 / self.space.ly)
        velocity_x = np.outer(ramp, profile)
        derivative_of_x = np.outer(ramp, profile_derivative)
        derivative_of_y = np.outer(slope, profile)
        return velocity_x, derivative_of_x, derivative_of_y, derivative_of_x

    def velocity(self, x: Array, y: Array) -> tuple[Array, Array]:
        """Values of (g_u, g_v), each shaped like meshgrid(x, y)."""
        velocity_x, _, _, _ = self.samples(x, y)
        return velocity_x, np.zeros_like(velocity_x)

    def derivatives(self, x: Array, y: Array) -> tuple[Array, Array, Array, Array]:
        """(d_x g_u, d_y g_u, d_x g_v, d_y g_v) sampled on the meshgrid, [y, x] shaped."""
        _, derivative_of_x, derivative_of_y, _ = self.samples(x, y)
        return derivative_of_x, derivative_of_y, np.zeros_like(derivative_of_x), np.zeros_like(derivative_of_x)

    def loads(self) -> tuple[Array, Array, Array]:
        """Momentum loads for both components and the continuity load, on nodes(extra).

        These are the right-hand side contributions that put the wall values into the
        equations; the matrix itself is unchanged.
        """
        x, _, y, _ = self.space.nodes(extra=self.extra)
        _, derivative_of_x, derivative_of_y, divergence = self.samples(x, y)
        load_x = self.space.stiffness_load(derivative_of_x, derivative_of_y)
        return load_x, np.zeros_like(load_x), self.space.divergence_load(divergence)

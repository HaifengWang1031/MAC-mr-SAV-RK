"""Dirichlet-combined Legendre bases and their exact one-dimensional matrices.

The velocity basis in one direction is the combination used by the shenfun
driven-cavity tutorial,

    phi_k(xi) = L_k(xi) - L_{k+2}(xi),    k = 0 .. size-1,   xi in [-1, 1],

which vanishes at both ends, so a tensor product of two such bases satisfies no-slip
on all four walls of the square. The pressure basis is the plain Legendre basis
truncated to the same index range, which is the inf-sup-safe choice the tutorial makes.

Every matrix here is built from two exact identities rather than from quadrature:

    int_{-1}^{1} L_k L_l dxi = 2/(2k+1) delta_kl
    L_m' = sum_{j in J_m} (2j+1) L_j     with J_m = (m-1, m-3, ... >= 0)

The second gives int L_m' L_n' dxi = 2 sum_{j in J_m ∩ J_n} (2j+1), an integer, and
int L_m' L_l dxi = 2 [l in J_m]. Mass entries are rational, so they are accumulated as
Fraction and converted once at the end: the matrices carry no quadrature error.

Physical coordinates on [0, length] come from x = (length/2)(xi+1), so every mass is
scaled by length/2, every derivative by 2/length, and a first derivative under the
integral by neither (the two cancel).
"""
from fractions import Fraction
from numpy.typing import NDArray
import numpy as np

Array = NDArray[np.float64]


def combination(k: int) -> tuple[tuple[int, int], ...]:
    """The Dirichlet combination phi_k = L_k - L_{k+2} as (degree, sign) pairs."""
    return ((k, 1), (k + 2, -1))


def derivative_indices(degree: int) -> list[int]:
    """J_m from L_m' = sum_{j in J_m} (2j+1) L_j."""
    return list(range(degree - 1, -1, -2))


def legendre_values(degrees: int, xi: Array) -> Array:
    """L_0..L_degrees at xi, leading axis over the degree."""
    out = np.empty((degrees + 1,) + np.shape(xi))
    out[0] = 1.
    if degrees >= 1:
        out[1] = xi
    for k in range(2, degrees + 1):
        out[k] = ((2 * k - 1) * xi * out[k - 1] - (k - 1) * out[k - 2]) / k
    return out


def legendre_derivatives(degrees: int, xi: Array) -> Array:
    """L_0'..L_degrees' at xi, from the derivative expansion; no 1/(1-xi**2) anywhere."""
    values = legendre_values(degrees, xi)
    out = np.zeros_like(values)
    for degree in range(1, degrees + 1):
        for index in derivative_indices(degree):
            out[degree] += (2 * index + 1) * values[index]
    return out


def dirichlet_values(size: int, xi: Array) -> Array:
    """phi_0..phi_{size-1} at xi."""
    values = legendre_values(size + 1, xi)
    return values[:size] - values[2:size + 2]


def dirichlet_derivatives(size: int, xi: Array) -> Array:
    """phi_0'..phi_{size-1}' at xi."""
    # P_{k+2}'-P_k'=(2k+3)P_{k+1}, including the endpoints.
    scale = (2 * np.arange(size) + 3).reshape((size,) + (1,) * np.ndim(xi))
    return -scale * legendre_values(size, xi)[1:size + 1]


def legendre_mass_diagonal(size: int) -> Array:
    """int L_m L_m dxi, the pressure-space Gram matrix (diagonal)."""
    return 2. / (2. * np.arange(size, dtype=float) + 1.)


def mass_reference(size: int) -> Array:
    """int phi_k phi_l dxi on the reference interval, exact."""
    out = np.zeros((size, size))
    for k in range(size):
        out[k, k] = float(Fraction(2, 2*k+1) + Fraction(2, 2*k+5))
        if k+2 < size:
            out[k, k+2] = out[k+2, k] = float(Fraction(-2, 2*k+5))
    return out


def stiffness_reference(size: int) -> Array:
    """int phi_k' phi_l' dxi on the reference interval, exact integer arithmetic."""
    return np.diag(2. * (2*np.arange(size) + 3))


def divergence_reference(size: int) -> Array:
    """int L_m phi_k' dxi = -int L_m' phi_k dxi, exact integers.

    Integration by parts drops the boundary term because phi_k vanishes at both ends.
    """
    out = np.zeros((size, size))
    for k in range(size-1):
        out[k+1, k] = -2.
    return out


def projection_reference(size: int) -> Array:
    """int L_m phi_k dxi, the pressure-to-velocity mass matrix, exact."""
    out = np.zeros((size, size))
    for k in range(size):
        out[k, k] = float(Fraction(2, 2*k+1))
        if k+2 < size:
            out[k+2, k] = float(Fraction(-2, 2*k+5))
    return out


def mass_scale(length: float) -> float:
    """Factor turning a reference-interval mass matrix into its physical counterpart."""
    return length / 2.


def stiffness_scale(length: float) -> float:
    """Factor turning a reference-interval stiffness matrix into its physical counterpart."""
    return 2. / length

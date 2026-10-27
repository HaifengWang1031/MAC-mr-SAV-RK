"""Exact tensor mass inverse using the two one-dimensional banded factors."""
import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

from . import basis
from .assembly import Array, Space


class TensorMass:
    def __init__(self, space: Space) -> None:
        self.size = space.size
        reference = basis.mass_reference(self.size)
        # Lower-band storage: diagonal in row 0, distance-two entries in row 2.
        band = np.zeros((3, self.size))
        band[0] = np.diag(reference)
        if self.size > 2:
            band[2, :-2] = np.diag(reference, k=-2)
        self.x = cholesky_banded(band * basis.mass_scale(space.lx), lower=True)
        self.y = cholesky_banded(band * basis.mass_scale(space.ly), lower=True)

    def solve(self, load: Array) -> Array:
        """Row-major [x mode,y mode] coefficients, without forming an inverse."""
        values = np.asarray(load).reshape(self.size, self.size)
        left = cho_solve_banded((self.x, True), values)
        result = cho_solve_banded((self.y, True), left.T).T
        return result.reshape(-1)

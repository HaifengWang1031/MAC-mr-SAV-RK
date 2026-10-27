"""Unpenalized shifted Stokes with an integral pressure gauge and full checks."""
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from .space import Array


@dataclass
class StokesSolution:
    velocity: Array
    pressure: Array
    residual: float
    continuity_residual: float


class FEMStokes:
    def __init__(self, mass_matrix: Any, stiffness: Any, divergence: Any, pressure_weights: Array,
                 *, tolerance: float = 1e-9, cache_size: int = 4) -> None:
        if not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError('Invalid solver controls')
        self.mass = mass_matrix
        self.stiffness = stiffness
        self.divergence = divergence
        self.pressure_weights = pressure_weights
        self.tolerance = tolerance
        self.cache_size = cache_size
        self.cache: OrderedDict[tuple[float, float], Any] = OrderedDict()

    def solve_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float) -> list[StokesSolution]:
        if not np.isfinite([mass, viscosity]).all() or min(mass, viscosity) < 0 or mass+viscosity <= 0:
            raise ValueError('Require nonnegative finite shifts with nonzero total')
        if not len(columns):
            return []
        nv = self.mass.shape[0]
        npres = self.divergence.shape[0]
        rhs = np.column_stack(columns)
        if rhs.shape[0] != nv or not np.isfinite(rhs).all():
            raise ValueError('Invalid velocity coefficients')
        load = self.mass @ rhs
        key = (float(mass), float(viscosity))
        if key not in self.cache:
            h = mass*self.mass + viscosity*self.stiffness
            c = sp.csc_matrix(self.pressure_weights[:, None])
            # The multiplier fixes only the pressure nullspace; no pressure penalty.
            matrix = sp.bmat([[h, -self.divergence.T, None],
                              [self.divergence, None, c],
                              [None, c.T, None]], format='csc')
            try:
                factor = splu(matrix)
            except RuntimeError as error:
                raise RuntimeError('Singular FEM Stokes system; check mesh resolution') from error
            self.cache[key] = factor
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        solved = self.cache[key].solve(np.vstack([load, np.zeros((npres+1, len(columns)))]))
        if not np.isfinite(solved).all():
            raise RuntimeError('Nonfinite FEM Stokes solution')
        velocity, pressure = solved[:nv], solved[nv:nv+npres]
        defect = mass*(self.mass @ velocity) + viscosity*(self.stiffness @ velocity) \
            - self.divergence.T @ pressure - load
        residuals = np.max(np.abs(defect), axis=0)/(1+np.max(np.abs(load), axis=0))
        # Check every pressure test row, independently of the gauge multiplier.
        continuity = np.max(np.abs(self.divergence @ velocity), axis=0)
        gauge = np.abs(self.pressure_weights @ pressure)
        results = []
        for k in range(len(columns)):
            if not np.isfinite([residuals[k], continuity[k], gauge[k]]).all() \
                    or residuals[k] > self.tolerance or continuity[k] > self.tolerance \
                    or gauge[k] > self.tolerance:
                raise RuntimeError(f'FEM Stokes residual={residuals[k]:.3e}, '
                                   f'continuity={continuity[k]:.3e}, pressure integral={gauge[k]:.3e}')
            results.append(StokesSolution(velocity[:, k].copy(), pressure[:, k].copy(),
                                          float(residuals[k]), float(continuity[k])))
        return results

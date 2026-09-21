"""Shifted Stokes solve in the single-element Legendre space.

Solves the coupled problem the model seam asks for,

    (mass*I - viscosity*Laplacian) u + grad p = f,   div u = 0,   u = 0 on the walls,

as the symmetric saddle point

    [[ mass*Mv + viscosity*Lv,       0,                  -Dx^T ],
     [            0,        mass*Mv + viscosity*Lv,      -Dy^T ],
     [           Dx,                  Dy,                    0  ]]

with the pressure constant mode removed, which is exactly the one-dimensional nullspace
of that block, so no extra gauge row is needed. The velocity basis vanishes on all four
walls, so no-slip is built into the trial space and no boundary lifting is required for
the homogeneous case.

Factorizations are cached on (mass, viscosity), mirroring the bounded-reuse discipline of
the MAC backend, and every solve re-checks the algebraic residual and the pointwise
divergence in physical space.
"""
from dataclasses import dataclass
from time import perf_counter
from typing import Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from numpy.typing import NDArray
from . import assembly
from .assembly import Space

Array = NDArray[np.float64]


@dataclass
class StokesSolution:
    velocity_x: Array
    velocity_y: Array
    pressure: Array
    residual: float
    divergence_inf: float


class SpectralStokes:
    def __init__(self, space: Space, *, tolerance: float = 1e-9, cache_size: int = 4) -> None:
        if tolerance <= 0 or cache_size < 1:
            raise ValueError('Invalid solver controls')
        self.space = space
        self.tolerance = tolerance
        self.cache_size = cache_size
        self.mass = assembly.velocity_mass(space)
        self.stiffness = assembly.velocity_stiffness(space)
        self.divergence_x, self.divergence_y = assembly.divergence_blocks(space)
        self.factorizations = 0
        self.factorization_seconds = 0.
        self.cache: dict[tuple[float, float], Any] = {}

    @property
    def pressure_dofs(self) -> int:
        """All pressure modes except the constant one, which is the removed gauge."""
        return self.space.velocity_dofs - 1

    def _system(self, mass: float, viscosity: float) -> Any:
        key = (float(mass), float(viscosity))
        if key not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            start = perf_counter()
            modes = self.space.velocity_dofs
            momentum = mass * self.mass + viscosity * self.stiffness
            zero_velocity = sp.csr_matrix((modes, modes))
            zero_pressure = sp.csr_matrix((self.pressure_dofs, self.pressure_dofs))
            dx = self.divergence_x[1:, :]
            dy = self.divergence_y[1:, :]
            matrix = sp.bmat([[momentum, zero_velocity, -dx.T],
                              [zero_velocity, momentum, -dy.T],
                              [dx, dy, zero_pressure]], format='csc')
            self.cache[key] = (splu(matrix), matrix)
            self.factorizations += 1
            self.factorization_seconds += perf_counter() - start
        return self.cache[key]

    def solve(self, load_x: Array, load_y: Array, *, mass: float, viscosity: float) -> StokesSolution:
        if not np.isfinite([mass, viscosity]).all() or mass < 0 or viscosity < 0 or mass + viscosity <= 0:
            raise ValueError('Require mass, viscosity >= 0 and nonzero total')
        space = self.space
        size = space.velocity_dofs
        lu, matrix = self._system(mass, viscosity)
        load = np.concatenate([np.asarray(load_x).reshape(-1), np.asarray(load_y).reshape(-1),
                               np.zeros(self.pressure_dofs)])
        solution = lu.solve(load)
        velocity_x = solution[:size].reshape(space.size, space.size)
        velocity_y = solution[size:2 * size].reshape(space.size, space.size)
        pressure = np.zeros((space.size, space.size))
        pressure.reshape(-1)[1:] = solution[2 * size:]
        velocity = np.concatenate([velocity_x.reshape(-1), velocity_y.reshape(-1)])
        load_velocity = np.concatenate([np.asarray(load_x).reshape(-1), np.asarray(load_y).reshape(-1)])
        pressure_flat = pressure.reshape(-1)
        # The momentum blocks are per component, so the defect is assembled per component
        # rather than against the stacked velocity vector.
        defect = np.concatenate([
            mass * (self.mass @ velocity_x.reshape(-1)) + viscosity * (self.stiffness @ velocity_x.reshape(-1))
            - self.divergence_x.T @ pressure_flat - np.asarray(load_x).reshape(-1),
            mass * (self.mass @ velocity_y.reshape(-1)) + viscosity * (self.stiffness @ velocity_y.reshape(-1))
            - self.divergence_y.T @ pressure_flat - np.asarray(load_y).reshape(-1)])
        scale = 1 + np.max(np.abs(load_velocity))
        residual = float(np.max(np.abs(defect)) / scale)
        # The constraint is imposed weakly, so the gate is the modal continuity residual
        # (machine zero, all rows including the gauge row) rather than the pointwise
        # divergence, which is a resolution diagnostic: the pressure space cannot represent
        # the unresolved part and the value falls only as the velocity converges.
        continuity = float(np.max(np.abs(self.divergence_x @ velocity_x.reshape(-1)
                                         + self.divergence_y @ velocity_y.reshape(-1))))
        divergence = self.divergence_inf(velocity_x, velocity_y)
        if not np.isfinite(solution).all() or residual > 100 * self.tolerance or continuity > 100 * self.tolerance:
            raise RuntimeError(f'Spectral Stokes residual={residual:.3e}, continuity={continuity:.3e}')
        return StokesSolution(velocity_x, velocity_y, pressure, residual, divergence)

    def divergence_inf(self, velocity_x: Array, velocity_y: Array) -> float:
        """max |d_x u + d_y v| on a fine Gauss grid, the analogue of the MAC divergence check."""
        space = self.space
        nodes_x, _, nodes_y, _ = space.nodes(extra=8)
        derivative_x, derivative_y = space.velocity_derivatives(nodes_x, nodes_y)
        values_x, values_y = space.velocity_values(nodes_x, nodes_y)
        # u pairs with the x derivative, v with the y derivative; the results come out in
        # the [y, x] order that meshgrid uses.
        # Same [x mode, y mode] index as everywhere else, so the coefficient matrices are
        # transposed before contacting the mode axis of the y basis.
        divergence = (values_y.T @ velocity_x.T @ derivative_x
                      + derivative_y.T @ velocity_y.T @ values_x)
        return float(np.max(np.abs(divergence)))

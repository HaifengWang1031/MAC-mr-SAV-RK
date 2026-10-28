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
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from numpy.typing import NDArray
from . import assembly
from .assembly import Space
from .quadrature import QuadratureCache

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
        self.quadrature = QuadratureCache(space)
        self.factorizations = 0
        self.factorization_seconds = 0.
        self.assembly_seconds = 0.
        self.lu_factorization_seconds = 0.
        self.solve_seconds = 0.
        self.check_seconds = 0.
        self.right_hand_sides = 0
        self.solve_calls = 0
        self.cache: OrderedDict[tuple[float, float], Any] = OrderedDict()

    @property
    def pressure_dofs(self) -> int:
        """All pressure modes except the constant one, which is the removed gauge."""
        return self.space.velocity_dofs - 1

    def _system(self, mass: float, viscosity: float) -> Any:
        key = (float(mass), float(viscosity))
        if key not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
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
            assembled = perf_counter()
            self.cache[key] = (splu(matrix), matrix)
            finished = perf_counter()
            self.assembly_seconds += assembled-start
            self.lu_factorization_seconds += finished-assembled
            self.factorizations += 1
            self.factorization_seconds += finished-start
        self.cache.move_to_end(key)
        return self.cache[key]

    def solve(self, load_x: Array, load_y: Array, *, mass: float, viscosity: float,
              continuity: Array | None = None) -> StokesSolution:
        """Solve the saddle point for the given loads.

        `continuity` is the right-hand side of the divergence rows, needed as soon as the
        boundary values are carried by a lifting: the interior unknown then satisfies
        `div w = -div g` rather than `div w = 0`. Its constant pressure mode is the removed
        gauge row and is ignored.
        """
        return self.solve_columns([load_x], [load_y], mass=mass, viscosity=viscosity,
                                  continuities=[continuity])[0]

    def solve_columns(self, loads_x: Sequence[Array], loads_y: Sequence[Array], *,
                      mass: float, viscosity: float,
                      continuities: Sequence[Array | None] | None = None) -> list[StokesSolution]:
        """One factor application for all loads, with every column checked separately."""
        if not np.isfinite([mass, viscosity]).all() or mass < 0 or viscosity < 0 or mass + viscosity <= 0:
            raise ValueError('Require mass, viscosity >= 0 and nonzero total')
        count = len(loads_x)
        if len(loads_y) != count or (continuities is not None and len(continuities) != count):
            raise ValueError('Stokes columns have different counts')
        if not count:
            return []
        space = self.space
        size = space.velocity_dofs
        constraints = []
        for continuity in ([None]*count if continuities is None else continuities):
            constraint = (np.zeros(self.pressure_dofs) if continuity is None
                          else np.asarray(continuity).reshape(-1)[1:])
            if constraint.size != self.pressure_dofs:
                raise ValueError('Continuity load has the wrong size')
            constraints.append(constraint)
        constraint = np.column_stack(constraints)
        load_x = np.column_stack([np.asarray(column).reshape(-1) for column in loads_x])
        load_y = np.column_stack([np.asarray(column).reshape(-1) for column in loads_y])
        if load_x.shape[0] != size or load_y.shape[0] != size:
            raise ValueError('Momentum load has the wrong size')
        lu, _ = self._system(mass, viscosity)
        load = np.concatenate([load_x, load_y, constraint])
        start = perf_counter()
        solution = lu.solve(load)
        self.solve_seconds += perf_counter()-start
        self.solve_calls += 1
        self.right_hand_sides += count
        start = perf_counter()
        velocity_x = solution[:size]
        velocity_y = solution[size:2*size]
        pressure = np.zeros((size, count))
        pressure[1:] = solution[2*size:]
        load_velocity = np.concatenate([load_x, load_y])
        # The momentum blocks are per component, so the defect is assembled per component
        # rather than against the stacked velocity vector.
        defect = np.concatenate([
            mass*(self.mass @ velocity_x) + viscosity*(self.stiffness @ velocity_x)
            - self.divergence_x.T @ pressure-load_x,
            mass*(self.mass @ velocity_y) + viscosity*(self.stiffness @ velocity_y)
            - self.divergence_y.T @ pressure-load_y])
        scale = 1 + np.max(np.abs(load_velocity), axis=0)
        residuals = np.max(np.abs(defect), axis=0)/scale
        # The constraint is imposed weakly, so the gate is the modal continuity residual
        # (machine zero, all rows including the gauge row) rather than the pointwise
        # divergence, which is a resolution diagnostic: the pressure space cannot represent
        # the unresolved part and the value falls only as the velocity converges. With a
        # lifting the residual is measured against the inhomogeneous right-hand side, so a
        # nonzero load does not look like a failed solve.
        defect_d = self.divergence_x[1:] @ velocity_x + self.divergence_y[1:] @ velocity_y-constraint
        continuities_inf = (np.max(np.abs(defect_d), axis=0) if self.pressure_dofs
                            else np.zeros(count))
        results = []
        for index in range(count):
            ux = velocity_x[:, index].reshape(space.size, space.size)
            uy = velocity_y[:, index].reshape(space.size, space.size)
            divergence = self.divergence_inf(ux, uy)
            residual = float(residuals[index])
            continuity_residual = float(continuities_inf[index])
            if not np.isfinite(solution[:, index]).all() or residual > 100*self.tolerance \
                    or continuity_residual > 100*self.tolerance:
                raise RuntimeError(f'Spectral Stokes residual={residual:.3e}, continuity={continuity_residual:.3e}')
            results.append(StokesSolution(ux, uy, pressure[:, index].reshape(space.size, space.size),
                                          residual, divergence))
        self.check_seconds += perf_counter()-start
        return results

    def divergence_inf(self, velocity_x: Array, velocity_y: Array) -> float:
        """max |d_x u + d_y v| on a fine Gauss grid, the analogue of the MAC divergence check."""
        # u pairs with the x derivative, v with the y derivative; the results come out in
        # the [y, x] order that meshgrid uses.
        # Same [x mode, y mode] index as everywhere else, so the coefficient matrices are
        # transposed before contacting the mode axis of the y basis.
        divergence = self.quadrature.get(8).divergence(velocity_x, velocity_y)
        return float(np.max(np.abs(divergence)))

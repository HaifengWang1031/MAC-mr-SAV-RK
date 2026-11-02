"""Consistent FEM weak loads for the shared time-stage algebra."""
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..model import SolveResult
from ..momentum import Momentum
from .model import FEMSolution
from .space import Array


class FEMMomentum(Momentum):
    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.fem = model
        self._boundary_load = np.asarray(model.full_linear[model.free_dofs] @ model._boundary_values)
        self._boundary_divergence = np.asarray(model.full_divergence @ model._boundary_values)

    def mass(self, velocity: Array) -> Array:
        return np.asarray(self.fem.mass @ self.fem._velocity(velocity))

    def linear(self, velocity: Array) -> Array:
        return np.asarray(self.fem.linear_stiffness @ self.fem._velocity(velocity))+self._boundary_load

    def gradient(self, pressure: Array) -> Array:
        return np.asarray(-self.fem.divergence.T @ self.fem._pressure(pressure))

    def force(self, t: float) -> Array:
        return self.mass(self.fem.force(t))

    def nonlinear(self, velocity: Array) -> Array:
        return self.nonlinear_with_lifting(velocity)[0]

    def nonlinear_with_lifting(self, velocity: Array) -> tuple[Array, float]:
        return self.fem.nonlinear_load_with_lifting(velocity)

    def pair(self, load: Array, velocity: Array) -> float:
        return float(load @ self.fem._velocity(velocity))

    def cauchy_scale(self, load: Array, velocity: Array) -> float:
        return float(np.sqrt(max(0., load @ self.fem._mass_factor.solve(load))*
                             self.fem.inner(velocity, velocity)))

    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> SolveResult:
        return self.solve_columns([rhs], mass=mass, viscosity=viscosity)[0]

    def solve_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float) -> list[SolveResult]:
        if not len(columns):
            return []
        loads = [self.fem._velocity(c) for c in columns]
        loads[0] = loads[0]-viscosity*self._boundary_load
        constraint = np.zeros((self.fem.pressure_basis.N, len(columns)))
        constraint[:, 0] = -self._boundary_divergence
        solutions = self.fem.stokes.solve_load_columns(loads, mass=mass, viscosity=viscosity,
                                                       divergence_rhs=constraint)
        results: list[SolveResult] = []
        for index, solution in enumerate(solutions):
            full = self.fem.full_velocity(solution.velocity) if index == 0 else np.zeros(self.fem.velocity_basis.N)
            if index:
                full[self.fem.free_dofs] = solution.velocity
            divergence = float(np.max(np.abs(self.fem._quadrature.divergence(full,
                                                    cache_field=len(solutions) == 1))))
            results.append(FEMSolution(solution.velocity, solution.pressure, solution.residual,
                                       divergence, solution.continuity_residual))
        return results

    def residual(self, defect: Array, rhs: Array, convection: Array | None = None, factor: float = 0.) -> float:
        # The numerator is the true consistent-mass coefficient defect. The
        # lower-bound denominator makes this stricter than the old criterion.
        denominator = 1+float(self.fem.stokes.load_norm_lower_bound(rhs))
        if convection is not None:
            denominator += factor*float(self.fem.stokes.load_norm_lower_bound(convection))
        return self.fem.max_abs(self.fem._mass_factor.solve(defect))/denominator

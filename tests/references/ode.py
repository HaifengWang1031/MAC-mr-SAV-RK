"""Known rotating ODE and controlled defects at the public Model/Trial seam.

The exact solution (cos(t), sin(t)) is specified independently of time stepping.
These fixtures do not import a test module, experiment configuration or runner.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from solver.core import Stage, Trial
from solver.mac.grid import Array


@dataclass
class State:
    t: float
    velocity: Array
    r: float = 0.


@dataclass
class Solution:
    velocity: Array
    pressure: Array
    residual: float = 0.
    divergence_inf: float = 0.


class ODEModel:
    nu = 1.
    L = np.diag([.7, 1.1])

    def vector(self, state: State) -> Array:
        return state.velocity

    def state(self, t: float, velocity: Array, r: float = 0.) -> State:
        return State(t, velocity, r)

    def combine(self, *terms: tuple[float, Array]) -> Array:
        return sum((weight*value for weight, value in terms), np.zeros_like(terms[0][1]))

    def apply_K(self, velocity: Array) -> Array:
        return self.L@velocity

    def apply_G(self, pressure: Array) -> Array:
        return np.zeros(2)

    def apply_D(self, velocity: Array) -> Array:
        return np.zeros(1)

    def physical_divergence_inf(self, velocity: Array) -> float:
        return 0.

    def inner(self, left: Array, right: Array) -> float:
        return float(left@right)

    def max_abs(self, field: Array) -> float:
        return float(np.max(np.abs(field)))

    def nonlinear(self, velocity: Array) -> Array:
        x, y = velocity
        return 3*x*np.array([-y, x])

    def nonlinear_with_lifting(self, velocity: Array) -> tuple[Array, float]:
        return self.nonlinear(velocity), 0.

    def exact(self, t: float) -> Array:
        return np.array([np.cos(t), np.sin(t)])

    def force(self, t: float) -> Array:
        u = self.exact(t)
        return np.array([-np.sin(t), np.cos(t)])+self.L@u+self.nonlinear(u)

    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> Solution:
        return Solution(np.linalg.solve(mass*np.eye(2)+viscosity*self.L, rhs), np.zeros(2))

    def solve_columns(self, columns: Sequence[Array], *, mass: float,
                      viscosity: float) -> list[Solution]:
        return [self.solve(col, mass=mass, viscosity=viscosity) for col in columns]

    def stage(self, pressure: Array, residual: float, divergence_inf: float,
              r: float = 0., candidates: Sequence[float] = (), root_residuals: Sequence[float] = (),
              scalar_residual: float = 0., *, continuity_residual: float = float('nan')) -> Stage:
        return Stage(pressure, residual, divergence_inf, r, list(candidates), list(root_residuals),
                     scalar_residual, continuity_residual)

    def diagnostics(self, state: State) -> dict[str, float]:
        return {'error': float(np.linalg.norm(state.velocity-self.exact(state.t)))}


class DefectScheme:
    """Known cubic velocity estimate and a first-order perturbed scalar update."""
    name = 'defect_test'

    def step(self, model: ODEModel, state: Any, dt: float) -> Trial:
        high = model.exact(state.t+dt)
        new_r = (state.r+dt**2)/(1+2*dt)
        # Trial accepts the Model's opaque state at runtime (also used by ODE tests).
        trial_state: Any = State(state.t+dt, high, new_r)
        return Trial(trial_state, [], high-np.array([dt**3, 0.]))

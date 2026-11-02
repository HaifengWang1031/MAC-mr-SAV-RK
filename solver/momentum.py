"""Stage momentum representation, independent of physical velocity storage.

The default adapter preserves the original Riesz/coefficient arithmetic. A FEM
model can supply weak loads instead without duplicating any RK or SAV formulas.
"""
from collections.abc import Sequence
from typing import Any

import numpy as np

from .model import Model, SolveResult


class Momentum:
    def __init__(self, model: Model) -> None:
        self.model = model

    def mass(self, velocity: Any) -> Any:
        return velocity

    def linear(self, velocity: Any) -> Any:
        return self.model.apply_K(velocity)

    def gradient(self, pressure: Any) -> Any:
        return self.model.apply_G(pressure)

    def force(self, t: float) -> Any:
        return self.model.force(t)

    def nonlinear(self, velocity: Any) -> Any:
        return self.model.nonlinear(velocity)

    def nonlinear_with_lifting(self, velocity: Any) -> tuple[Any, float]:
        return self.model.nonlinear_with_lifting(velocity)

    def pair(self, load: Any, velocity: Any) -> float:
        return self.model.inner(load, velocity)

    def cauchy_scale(self, load: Any, velocity: Any) -> float:
        return float(np.sqrt(self.model.inner(load, load)*self.model.inner(velocity, velocity)))

    def solve(self, rhs: Any, *, mass: float, viscosity: float) -> SolveResult:
        return self.model.solve(rhs, mass=mass, viscosity=viscosity)

    def solve_columns(self, columns: Sequence[Any], *, mass: float, viscosity: float) -> list[SolveResult]:
        return self.model.solve_columns(columns, mass=mass, viscosity=viscosity)

    def residual(self, defect: Any, rhs: Any, convection: Any = None, factor: float = 0.) -> float:
        denominator = 1+self.model.max_abs(rhs)
        if convection is not None:
            denominator += factor*self.model.max_abs(convection)
        return self.model.max_abs(defect)/denominator


def stage_momentum(model: Model) -> Momentum:
    factory = getattr(model, 'stage_momentum', None)
    return Momentum(model) if factory is None else factory()

"""Model-owned, bounded quadrature plans; public Space sampling stays uncached."""
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .assembly import Array, Space


@dataclass(frozen=True)
class QuadraturePlan:
    x: Array
    y: Array
    weight: Array
    values_x: Array
    values_y: Array
    derivative_x: Array
    derivative_y: Array
    pressure_x: Array
    pressure_y: Array
    weighted_x: Array
    weighted_y: Array

    @classmethod
    def build(cls, space: Space, extra: int) -> "QuadraturePlan":
        x, wx, y, wy = space.nodes(extra=extra)
        vx, vy = space.velocity_values(x, y)
        dx, dy = space.velocity_derivatives(x, y)
        px, py = space.pressure_values(x, y)
        arrays = (x, y, np.outer(wy, wx), vx, vy, dx, dy, px, py,
                  vx * wx[None, :], vy * wy[None, :])
        for array in arrays:
            array.flags.writeable = False
        return cls(*arrays)

    def evaluate(self, coefficients: Array) -> Array:
        return self.values_y.T @ coefficients.T @ self.values_x

    def sample(self, coefficients: Array) -> tuple[Array, Array, Array]:
        """Value and both derivatives, sharing the first tensor contraction."""
        left = self.values_y.T @ coefficients.T
        derivative_left = self.derivative_y.T @ coefficients.T
        return left @ self.values_x, left @ self.derivative_x, derivative_left @ self.values_x

    def project(self, values: Array) -> Array:
        return self.weighted_x @ values.T @ self.weighted_y.T

    def stiffness_load(self, flux_x: Array, flux_y: Array) -> Array:
        first = self.values_y @ (flux_x * self.weight) @ self.derivative_x.T
        second = self.derivative_y @ (flux_y * self.weight) @ self.values_x.T
        return first.T + second.T

    def divergence(self, u: Array, v: Array) -> Array:
        return (self.values_y.T @ u.T @ self.derivative_x
                + self.derivative_y.T @ v.T @ self.values_x)


class QuadratureCache:
    """One immutable Space per cache; only a few quadrature resolutions are retained."""

    def __init__(self, space: Space, capacity: int = 4) -> None:
        self.space = space
        self.capacity = capacity
        self.plans: OrderedDict[int, QuadraturePlan] = OrderedDict()

    def get(self, extra: int) -> QuadraturePlan:
        if extra not in self.plans:
            plan = QuadraturePlan.build(self.space, extra)
            if len(self.plans) >= self.capacity:
                self.plans.popitem(last=False)
            self.plans[extra] = plan
        self.plans.move_to_end(extra)
        return self.plans[extra]

    def clear(self) -> None:
        self.plans.clear()

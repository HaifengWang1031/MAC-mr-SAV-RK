"""First-order IMEX Euler: explicit convection/force, implicit constrained Stokes."""
from typing import Any
import numpy as np

from ..core import Trial
from ..model import Model, stage_divergences
from ..momentum import stage_momentum


class IMEXEuler:
    name = 'imex_euler'

    def step(self, model: Model, state: Any, dt: float) -> Trial:
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(state.t):
            raise ValueError('Invalid step size or state time')
        momentum = stage_momentum(model)
        old = model.vector(state)
        convection = momentum.nonlinear(old)
        rhs = model.combine((1., momentum.mass(old)), (dt, momentum.force(state.t)), (-dt, convection))
        solved = momentum.solve(rhs, mass=1., viscosity=model.nu*dt)
        velocity = solved.velocity
        pressure = model.combine((1/dt, solved.pressure))
        defect = model.combine((1., momentum.mass(velocity)), (model.nu*dt, momentum.linear(velocity)),
                               (dt, momentum.gradient(pressure)), (-1., rhs))
        residual = momentum.residual(defect, rhs)
        physical_divergence, continuity = stage_divergences(model, velocity)
        if not np.isfinite([residual, physical_divergence, continuity,
                            model.max_abs(velocity), model.max_abs(pressure)]).all() \
                or residual > 1e-8 or continuity > 1e-8*(1+model.max_abs(velocity)):
            raise RuntimeError(f'IMEX Euler residual {residual}, divergence {continuity}')
        stage = model.stage(pressure, residual, physical_divergence, continuity_residual=continuity)
        return Trial(model.state(state.t+dt, velocity), [stage])

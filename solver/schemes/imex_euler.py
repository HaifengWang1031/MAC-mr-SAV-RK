"""First-order IMEX Euler: explicit convection/force, implicit constrained Stokes."""
from typing import Any
import numpy as np

from ..core import Trial
from ..model import Model, stage_divergences


class IMEXEuler:
    name = 'imex_euler'

    def step(self, model: Model, state: Any, dt: float) -> Trial:
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(state.t):
            raise ValueError('Invalid step size or state time')
        old = model.vector(state)
        convection = model.nonlinear(old)
        rhs = model.combine((1., old), (dt, model.force(state.t)), (-dt, convection))
        solved = model.solve(rhs, mass=1., viscosity=model.nu*dt)
        velocity = solved.velocity
        pressure = model.combine((1/dt, solved.pressure))
        defect = model.combine((1., velocity), (model.nu*dt, model.apply_K(velocity)),
                               (dt, model.apply_G(pressure)), (-1., rhs))
        residual = model.max_abs(defect)/(1+model.max_abs(rhs))
        physical_divergence, continuity = stage_divergences(model, velocity)
        if not np.isfinite([residual, physical_divergence, continuity,
                            model.max_abs(velocity), model.max_abs(pressure)]).all() \
                or residual > 1e-8 or continuity > 1e-8*(1+model.max_abs(velocity)):
            raise RuntimeError(f'IMEX Euler residual {residual}, divergence {continuity}')
        stage = model.stage(pressure, residual, physical_divergence, continuity_residual=continuity)
        return Trial(model.state(state.t+dt, velocity), [stage])

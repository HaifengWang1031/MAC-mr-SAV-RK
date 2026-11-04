"""First-order IMEX Euler-mr-ccSAV with one frozen scalar correction, Q=1."""
from typing import Any
import numpy as np

from ..core import Trial
from ..model import Model, stage_divergences
from ..momentum import stage_momentum
from .scalar_correction import solve_scalar_stage


class IMEXEulerMRSAV:
    name = 'imex_euler_mrsav'

    def __init__(self, gamma: float = 1.) -> None:
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError('gamma must be finite and nonnegative')
        self.gamma = gamma

    def step(self, model: Model, state: Any, dt: float) -> Trial:
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite([state.t, state.r]).all():
            raise ValueError('Invalid step size or scalar state')
        momentum = stage_momentum(model)
        old = model.vector(state)
        convection, lifting_work = momentum.nonlinear_with_lifting(old)
        rhs = model.combine((1., momentum.mass(old)), (dt, momentum.force(state.t)))
        solved = momentum.solve_columns([rhs, convection], mass=1., viscosity=model.nu*dt)
        V, W = solved[0].velocity, solved[1].velocity
        alpha = momentum.pair(convection, V)+lifting_work
        beta = momentum.pair(convection, W)
        if beta < 0.:
            scale = momentum.cauchy_scale(convection, W)
            if beta < -1e-12*(1+scale):
                raise RuntimeError('Negative Stokes response energy')
        beta = max(beta, 0.)
        scalar_mass = 1+self.gamma*dt
        scalar_update = solve_scalar_stage(state.r, dt, scalar_mass, state.r, alpha, beta, order=1)
        r, correction = scalar_update.r, scalar_update.coupling
        velocity = model.combine((1., V), (-dt*correction, W))
        pressure = model.combine((1/dt, solved[0].pressure), (-correction, solved[1].pressure))
        scalar = scalar_mass*r-state.r+dt*(momentum.pair(convection, velocity)+lifting_work)
        scalar_residual = abs(scalar)/(1+abs(state.r)+abs(r))
        defect = model.combine((1., momentum.mass(velocity)), (model.nu*dt, momentum.linear(velocity)),
                               (dt, momentum.gradient(pressure)), (-1., rhs), (dt*correction, convection))
        residual = momentum.residual(defect, rhs, convection, dt*abs(correction))
        physical_divergence, continuity = stage_divergences(model, velocity)
        if not np.isfinite([r, residual, scalar_residual, physical_divergence, continuity,
                            model.max_abs(velocity), model.max_abs(pressure)]).all() \
                or residual > 1e-8 or scalar_residual > 1e-8 \
                or continuity > 1e-8*(1+model.max_abs(velocity)):
            raise RuntimeError(f'IMEX Euler SAV residuals {residual}, {scalar_residual}, divergence {continuity}')
        stage = scalar_update.record(model.stage(pressure, residual, physical_divergence, r,
                            scalar_residual=scalar_residual, continuity_residual=continuity))
        return Trial(model.state(state.t+dt, velocity, r), [stage])

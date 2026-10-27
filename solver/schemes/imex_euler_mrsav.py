"""First-order IMEX Euler-mr-ccSAV with G=1-r, Q=1 and a linear scalar solve."""
from typing import Any
import numpy as np

from ..core import Trial
from ..model import Model, stage_divergences


class IMEXEulerMRSAV:
    name = 'imex_euler_mrsav'

    def __init__(self, gamma: float = 1.) -> None:
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError('gamma must be finite and nonnegative')
        self.gamma = gamma

    def step(self, model: Model, state: Any, dt: float) -> Trial:
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite([state.t, state.r]).all():
            raise ValueError('Invalid step size or scalar state')
        old = model.vector(state)
        convection, lifting_work = model.nonlinear_with_lifting(old)
        rhs = model.combine((1., old), (dt, model.force(state.t)))
        solved = model.solve_columns([rhs, convection], mass=1., viscosity=model.nu*dt)
        V, W = solved[0].velocity, solved[1].velocity
        alpha = model.inner(convection, V)+lifting_work
        beta = model.inner(convection, W)
        if beta < 0.:
            scale = np.sqrt(model.inner(convection, convection)*model.inner(W, W))
            if beta < -1e-12*(1+scale):
                raise RuntimeError('Negative Stokes response energy')
        beta = max(beta, 0.)
        scalar_mass = 1+self.gamma*dt
        denominator = scalar_mass+dt*dt*beta
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError('Invalid IMEX Euler SAV scalar denominator')
        r = (state.r-dt*alpha+dt*dt*beta)/denominator
        correction = 1-r
        velocity = model.combine((1., V), (-dt*correction, W))
        pressure = model.combine((1/dt, solved[0].pressure), (-correction, solved[1].pressure))
        scalar = scalar_mass*r-state.r+dt*(model.inner(convection, velocity)+lifting_work)
        scalar_residual = abs(scalar)/(1+abs(state.r)+abs(r))
        defect = model.combine((1., velocity), (model.nu*dt, model.apply_K(velocity)),
                               (dt, model.apply_G(pressure)), (-1., rhs), (dt*correction, convection))
        residual = model.max_abs(defect)/(1+model.max_abs(rhs)+dt*abs(correction)*model.max_abs(convection))
        physical_divergence, continuity = stage_divergences(model, velocity)
        if not np.isfinite([r, residual, scalar_residual, physical_divergence, continuity,
                            model.max_abs(velocity), model.max_abs(pressure)]).all() \
                or residual > 1e-8 or scalar_residual > 1e-8 \
                or continuity > 1e-8*(1+model.max_abs(velocity)):
            raise RuntimeError(f'IMEX Euler SAV residuals {residual}, {scalar_residual}, divergence {continuity}')
        # The sole candidate is the directly computed root of a linear equation.
        stage = model.stage(pressure, residual, physical_divergence, r, [r], [scalar_residual],
                            scalar_residual, continuity_residual=continuity)
        return Trial(model.state(state.t+dt, velocity, r), [stage])

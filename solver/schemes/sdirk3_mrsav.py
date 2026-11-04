"""Four-stage SDIRK3-mr-ccSAV with exactly two frozen scalar corrections per stage."""
from typing import Any
import numpy as np
from ..core import Trial
from ..model import Model, stage_divergences
from ..momentum import stage_momentum
from .scalar_correction import solve_scalar_stage
from .sdirk3 import C, D, known_terms, embedded_velocity


class SDIRK3MRSAV:
    name = 'sdirk3_mrsav'

    def __init__(self, gamma: float = 1.) -> None:
        if not np.isfinite(gamma) or gamma < 0: raise ValueError('gamma must be finite and nonnegative')
        self.gamma = gamma

    def step(self, model: Model, state: Any, dt: float) -> Trial:
        if not np.isfinite(dt) or dt <= 0: raise ValueError('Invalid step size')
        momentum = stage_momentum(model)
        velocities = [model.vector(state)]
        linear: list[Any] = []
        scalars = [state.r]
        stages = []
        nonlinear = [momentum.nonlinear_with_lifting(velocities[0])]
        forces = [momentum.force(state.t+C[j]*dt) for j in range(4)]
        for i in range(4):
            convection, force, lifting = known_terms(model, nonlinear, forces, i)
            rhs = model.combine((1., momentum.mass(velocities[-1])), (dt, force),
                                *[(-model.nu*dt*D[i,j], linear[j])
                                  for j in range(i)])
            scalar_rhs = scalars[-1]-self.gamma*dt*sum(D[i,j]*scalars[j+1] for j in range(i))
            solved = momentum.solve_columns([rhs, convection], mass=1., viscosity=model.nu*dt*D[i,i])
            V, W = solved[0].velocity, solved[1].velocity
            alpha = momentum.pair(convection, V)+lifting
            beta = momentum.pair(convection, W)
            if beta < 0.:
                scale = momentum.cauchy_scale(convection,W)
                if beta < -1e-12*(1+scale):
                    raise RuntimeError('Negative Stokes response energy')
            beta = max(beta, 0.)
            scalar_update = solve_scalar_stage(scalars[-1],dt,1+self.gamma*dt*D[i,i],
                                               scalar_rhs,alpha,beta,order=3)
            r, coupling = scalar_update.r, scalar_update.coupling
            response = dt*coupling
            velocity = model.combine((1., V), (-response, W))
            pressure = model.combine((1/dt, solved[0].pressure), (-coupling, solved[1].pressure))
            scalar = (1+self.gamma*dt*D[i,i])*r-scalar_rhs + dt*scalar_update.factor*(momentum.pair(convection,velocity)+lifting)
            scalar_residual = abs(scalar)/(1+abs(scalar_rhs)+abs(r))
            linear_velocity = momentum.linear(velocity)
            defect = model.combine((1.,momentum.mass(velocity)), (model.nu*dt*D[i,i],linear_velocity),
                                   (dt,momentum.gradient(pressure)), (-1.,rhs), (dt*coupling,convection))
            residual = momentum.residual(defect, rhs, convection, dt*abs(coupling))
            physical_divergence,continuity = stage_divergences(model,velocity)
            if not np.isfinite([residual,scalar_residual,continuity]).all() or residual>1e-8 \
                    or scalar_residual>1e-8 or continuity>1e-8*(1+model.max_abs(velocity)):
                raise RuntimeError(f'SAV stage residuals {residual}, {scalar_residual}, divergence {continuity}')
            velocities.append(velocity)
            linear.append(linear_velocity)
            if i < 3: nonlinear.append(momentum.nonlinear_with_lifting(velocity))
            scalars.append(r)
            stages.append(scalar_update.record(model.stage(pressure, residual, physical_divergence, r,
                                      scalar_residual=scalar_residual, continuity_residual=continuity)))
        return Trial(model.state(state.t+dt, velocities[-1], scalars[-1]), stages,
                     embedded_velocity(model, velocities))

"""Four-stage incremental SDIRK3-mr-ccSAV, G=1-r^3 and Q=1+r+r^2."""
from typing import Any
import numpy as np
from numpy.polynomial import Polynomial
from ..core import Trial
from ..model import Model
from .roots import real_roots
from .sdirk3 import C, D, known_terms, embedded_velocity


class SDIRK3MRSAV:
    name = 'sdirk3_mrsav'

    def __init__(self, gamma: float = 1.) -> None:
        if not np.isfinite(gamma) or gamma < 0: raise ValueError('gamma must be finite and nonnegative')
        self.gamma = gamma

    def step(self, model: Model, state: Any, dt: float) -> Trial:
        if not np.isfinite(dt) or dt <= 0: raise ValueError('Invalid step size')
        velocities = [model.vector(state)]
        scalars = [state.r]
        stages = []
        nonlinear = [model.nonlinear_with_lifting(velocities[0])]
        forces = [model.force(state.t+C[j]*dt) for j in range(4)]
        for i in range(4):
            convection, force, lifting = known_terms(model, nonlinear, forces, i)
            rhs = model.combine((1., velocities[-1]), (dt, force),
                                *[(-model.nu*dt*D[i,j], model.apply_K(velocities[j+1]))
                                  for j in range(i)])
            scalar_rhs = scalars[-1]-self.gamma*dt*sum(D[i,j]*scalars[j+1] for j in range(i))
            solved = model.solve_columns([rhs, convection], mass=1., viscosity=model.nu*dt*D[i,i])
            V, W = solved[0].velocity, solved[1].velocity
            alpha = model.inner(convection, V)+lifting
            beta = model.inner(convection, W)
            if beta < -1e-12*(1+np.sqrt(model.inner(convection,convection)*model.inner(W,W))):
                raise RuntimeError('Negative Stokes response energy')
            beta = max(beta, 0.)
            rpoly = Polynomial([0.,1.])
            qpoly = Polynomial([1.,1.,1.])
            gpoly = Polynomial([1.,0.,0.,-1.])
            poly = (1+self.gamma*dt*D[i,i])*rpoly-scalar_rhs + dt*alpha*qpoly-dt*dt*beta*qpoly*gpoly
            choice = real_roots(poly.coef[::-1])
            r = choice.selected
            velocity = model.combine((1., V), (-dt*(1-r**3), W))
            pressure = model.combine((1/dt, solved[0].pressure), (-(1-r**3), solved[1].pressure))
            scalar = (1+self.gamma*dt*D[i,i])*r-scalar_rhs + dt*(1+r+r*r)*(model.inner(convection,velocity)+lifting)
            scalar_residual = abs(scalar)/(1+abs(scalar_rhs)+abs(r))
            defect = model.combine((1.,velocity), (model.nu*dt*D[i,i],model.apply_K(velocity)),
                                   (dt,model.apply_G(pressure)), (-1.,rhs), (dt*(1-r**3),convection))
            residual = model.max_abs(defect)/(1+model.max_abs(rhs)+dt*abs(1-r**3)*model.max_abs(convection))
            continuity = model.max_abs(model.apply_D(velocity))
            if not np.isfinite([residual,scalar_residual,continuity]).all() or residual>1e-8 \
                    or scalar_residual>1e-8 or continuity>1e-8*(1+model.max_abs(velocity)):
                raise RuntimeError(f'SAV stage residuals {residual}, {scalar_residual}, divergence {continuity}')
            velocities.append(velocity)
            if i < 3: nonlinear.append(model.nonlinear_with_lifting(velocity))
            scalars.append(r)
            stages.append(model.stage(pressure, residual, model.physical_divergence_inf(velocity), r,
                                      choice.candidates, choice.residuals, scalar_residual,
                                      continuity_residual=continuity))
        return Trial(model.state(state.t+dt, velocities[-1], scalars[-1]), stages,
                     embedded_velocity(model, velocities))

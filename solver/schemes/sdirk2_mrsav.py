"""Incremental SDIRK2-mr-ccSAV with one frozen-factor correction per stage.

Written against the model seam (`solver/model.py`) like SDIRK2, so both schemes run on
every discretisation that supplies the seam.
"""
from typing import Any
import numpy as np
from ..core import Trial
from ..model import Model, stage_divergences
from ..momentum import stage_momentum
from .sdirk2 import ETA, DELTA
from .scalar_correction import solve_scalar_stage

class SDIRK2MRSAV:
    name='sdirk2_mrsav'

    def __init__(self, gamma: float = 1.) -> None:
        if not np.isfinite(gamma) or gamma<0: raise ValueError('gamma must be finite and nonnegative')
        self.gamma=gamma

    def _stage(self, model: Model, rhs: Any, scalar_rhs: float, convection: Any,
               dt: float, predictor: float, lifting_work: float = 0.) -> tuple[Any, Any, Any]:
        # Two right-hand sides sharing one matrix: the model may batch them and split
        # the columns (bit-exact) or loop, whichever its backend supports.
        momentum = stage_momentum(model)
        solved=momentum.solve_columns([rhs,convection],mass=1.,viscosity=model.nu*ETA*dt)
        V,W=solved[0].velocity,solved[1].velocity
        alpha=momentum.pair(convection,V)+lifting_work
        beta=momentum.pair(convection,W)
        # Same guard as the earlier explicit-norm form, written with inner products so
        # the seam needs no separate norm member: |B|*|W| = sqrt(<B,B>*<W,W>).
        if beta < 0.:
            scale=momentum.cauchy_scale(convection,W)
            if beta < -1e-12*(1+scale):
                raise RuntimeError('Negative Stokes response energy')
        beta=max(beta,0.)
        scalar_update=solve_scalar_stage(predictor,dt,1+self.gamma*ETA*dt,scalar_rhs,alpha,beta,order=2)
        r=scalar_update.r
        coupling=scalar_update.coupling
        response=dt*coupling
        velocity=model.combine((1.,V),(-response,W))
        pressure=model.combine((1./dt,solved[0].pressure),(-coupling,solved[1].pressure))
        scalar=(1+self.gamma*ETA*dt)*r-scalar_rhs+dt*scalar_update.factor*(momentum.pair(convection,velocity)+lifting_work)
        scalar_residual=abs(scalar)/(1+abs(scalar_rhs)+abs(r))
        linear_velocity=momentum.linear(velocity)
        defect=model.combine((1.,momentum.mass(velocity)),(model.nu*ETA*dt,linear_velocity),
                             (dt,momentum.gradient(pressure)),(-1.,rhs),(dt*coupling,convection))
        residual=momentum.residual(defect, rhs, convection, dt*abs(coupling))
        physical_divergence,divergence=stage_divergences(model,velocity)
        # A non-finite field value propagates into these three scalars, so they cover the
        # earlier explicit finiteness check on the velocity array.
        if not np.isfinite([residual,scalar_residual,divergence]).all() or scalar_residual>1e-8 \
                or residual>1e-8 or divergence>1e-8*(1+model.max_abs(velocity)):
            # All three are reported: the divergence is the one that fails whenever the linear
            # tolerance is loose, and omitting it made earlier failures look like residual ones.
            raise RuntimeError(f'SAV stage residuals {residual}, {scalar_residual}, divergence {divergence}')
        stage=scalar_update.record(model.stage(pressure,residual,physical_divergence,r,
                                    scalar_residual=scalar_residual,continuity_residual=divergence))
        return velocity,stage,linear_velocity

    def step(self, model: Model, state: Any, dt: float) -> Trial:
        if not np.isfinite(dt) or dt<=0: raise ValueError('Invalid step size')
        momentum = stage_momentum(model)
        old=model.vector(state)
        nonlinear_old,lifting_old=momentum.nonlinear_with_lifting(old)
        f0=momentum.force(state.t)
        first,s1,k1=self._stage(model,model.combine((1.,momentum.mass(old)),(ETA*dt,f0)),state.r,
                                model.combine((ETA,nonlinear_old)),dt,state.r,ETA*lifting_old)
        n1,lifting_first=momentum.nonlinear_with_lifting(first);f1=momentum.force(state.t+ETA*dt)
        rhs=model.combine((1.,momentum.mass(first)),(-model.nu*dt*(1-2*ETA),k1),(-dt,f0),(dt*(1-DELTA),f1))
        second,s2,_=self._stage(model,rhs,(1-self.gamma*dt*(1-2*ETA))*s1.r,
                                model.combine((-1.,nonlinear_old),(1-DELTA,n1)),dt,s1.r,
                                -lifting_old+(1-DELTA)*lifting_first)
        embedded=model.combine((1.-1./ETA,old),(1./ETA,first))
        return Trial(model.state(state.t+dt,second,s2.r),[s1,s2],embedded)

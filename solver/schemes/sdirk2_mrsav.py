"""The notes' incremental SDIRK2-mr-ccSAV, Gu=1-r^2, Gr=1+r.

Written against the model seam (`solver/model.py`) like SDIRK2, so both schemes run on
every discretisation that supplies the seam.
"""
from typing import Any
import numpy as np
from ..core import Trial
from ..model import Model
from .sdirk2 import ETA, DELTA
from .roots import real_roots

class SDIRK2MRSAV:
    name='sdirk2_mrsav'

    def __init__(self, gamma: float = 1.) -> None:
        if not np.isfinite(gamma) or gamma<0: raise ValueError('gamma must be finite and nonnegative')
        self.gamma=gamma

    def _stage(self, model: Model, rhs: Any, scalar_rhs: float, convection: Any,
               dt: float, lifting_work: float = 0.) -> tuple[Any, Any]:
        # Two right-hand sides sharing one matrix: the model may batch them and split
        # the columns (bit-exact) or loop, whichever its backend supports.
        solved=model.solve_columns([rhs,convection],mass=1.,viscosity=model.nu*ETA*dt)
        V,W=solved[0].velocity,solved[1].velocity
        alpha=model.inner(convection,V)+lifting_work
        beta=model.inner(convection,W)
        # Same guard as the earlier explicit-norm form, written with inner products so
        # the seam needs no separate norm member: |B|*|W| = sqrt(<B,B>*<W,W>).
        if beta < -1e-12*(1+np.sqrt(model.inner(convection,convection)*model.inner(W,W))):
            raise RuntimeError('Negative Stokes response energy')
        beta=max(beta,0.)
        cubic=np.array([dt*dt*beta,dt*dt*beta,1+self.gamma*ETA*dt+dt*alpha-dt*dt*beta,
                        -scalar_rhs+dt*alpha-dt*dt*beta])
        choice=real_roots(cubic)
        r=choice.selected
        velocity=model.combine((1.,V),(-dt*(1-r*r),W))
        pressure=model.combine((1./dt,solved[0].pressure),(-(1-r*r),solved[1].pressure))
        scalar=(1+self.gamma*ETA*dt)*r-scalar_rhs+dt*(1+r)*(model.inner(convection,velocity)+lifting_work)
        scalar_residual=abs(scalar)/(1+abs(scalar_rhs)+abs(r))
        defect=model.combine((1.,velocity),(model.nu*ETA*dt,model.apply_K(velocity)),
                             (dt,model.apply_G(pressure)),(-1.,rhs),(dt*(1-r*r),convection))
        residual=model.max_abs(defect)/(1+model.max_abs(rhs)+dt*abs(1-r*r)*model.max_abs(convection))
        divergence=model.max_abs(model.apply_D(velocity))
        # A non-finite field value propagates into these three scalars, so they cover the
        # earlier explicit finiteness check on the velocity array.
        if not np.isfinite([residual,scalar_residual,divergence]).all() or scalar_residual>1e-8 \
                or residual>1e-8 or divergence>1e-8*(1+model.max_abs(velocity)):
            # All three are reported: the divergence is the one that fails whenever the linear
            # tolerance is loose, and omitting it made earlier failures look like residual ones.
            raise RuntimeError(f'SAV stage residuals {residual}, {scalar_residual}, divergence {divergence}')
        return velocity,model.stage(pressure,residual,divergence,r,choice.candidates,
                                    choice.residuals,scalar_residual)

    def step(self, model: Model, state: Any, dt: float) -> Trial:
        if not np.isfinite(dt) or dt<=0: raise ValueError('Invalid step size')
        old=model.vector(state)
        nonlinear_old,lifting_old=model.nonlinear_with_lifting(old)
        f0=model.force(state.t)
        first,s1=self._stage(model,model.combine((1.,old),(ETA*dt,f0)),state.r,
                             model.combine((ETA,nonlinear_old)),dt,ETA*lifting_old)
        k1=model.apply_K(first);n1,lifting_first=model.nonlinear_with_lifting(first);f1=model.force(state.t+ETA*dt)
        rhs=model.combine((1.,first),(-model.nu*dt*(1-2*ETA),k1),(-dt,f0),(dt*(1-DELTA),f1))
        second,s2=self._stage(model,rhs,(1-self.gamma*dt*(1-2*ETA))*s1.r,
                              model.combine((-1.,nonlinear_old),(1-DELTA,n1)),dt,
                              -lifting_old+(1-DELTA)*lifting_first)
        return Trial(model.state(state.t+dt,second,s2.r),[s1,s2])

"""The notes' incremental SDIRK2-mr-ccSAV, Gu=1-r², Gr=1+r."""
import numpy as np
from ..core import State, Stage, Trial
from ..mac.grid import Array
from ..mac_ns import MACNavierStokes
from .sdirk2 import ETA, DELTA
from .roots import real_roots

class SDIRK2MRSAV:
    name='sdirk2_mrsav'

    def __init__(self, gamma: float = 1.) -> None:
        if not np.isfinite(gamma) or gamma<0: raise ValueError('gamma must be finite and nonnegative')
        self.gamma=gamma

    def _stage(self, model: MACNavierStokes, rhs: Array, scalar_rhs: float,
               convection: Array, dt: float) -> tuple[Array,Stage]:
        solved=model.backend.solve(np.column_stack((rhs,convection)),mass=1.,viscosity=model.nu*ETA*dt)
        V,W=solved.velocity[:,0],solved.velocity[:,1]
        alpha=model.grid.inner(convection,V)
        beta=model.grid.inner(convection,W)
        if beta < -1e-12*(1+model.grid.norm(convection)*model.grid.norm(W)):
            raise RuntimeError('Negative Stokes response energy')
        beta=max(beta,0.)
        cubic=np.array([dt*dt*beta,dt*dt*beta,1+self.gamma*ETA*dt+dt*alpha-dt*dt*beta,
                        -scalar_rhs+dt*alpha-dt*dt*beta])
        choice=real_roots(cubic)
        r=choice.selected
        velocity=V-dt*(1-r*r)*W
        pressure=(solved.pressure[:,:,0]-dt*(1-r*r)*solved.pressure[:,:,1])/dt
        scalar=(1+self.gamma*ETA*dt)*r-scalar_rhs+dt*(1+r)*model.grid.inner(convection,velocity)
        scalar_residual=abs(scalar)/(1+abs(scalar_rhs)+abs(r))
        defect=velocity+model.nu*ETA*dt*(model.ops.K@velocity)+dt*(model.ops.G@pressure.ravel())-rhs+dt*(1-r*r)*convection
        residual=float(np.max(np.abs(defect))/(1+np.max(np.abs(rhs))+dt*abs(1-r*r)*np.max(np.abs(convection))))
        divergence=float(np.max(np.abs(model.ops.D@velocity)))
        if not np.isfinite(velocity).all() or scalar_residual>1e-8 or residual>1e-8 or divergence>1e-8*(1+np.max(np.abs(velocity))):
            raise RuntimeError(f'SAV stage residuals {residual}, {scalar_residual}')
        return velocity,Stage(pressure,residual,divergence,r,choice.candidates,choice.residuals,scalar_residual)

    def step(self, model: MACNavierStokes, state: State, dt: float) -> Trial:
        if not np.isfinite(dt) or dt<=0: raise ValueError('Invalid step size')
        old=model.vector(state)
        nonlinear_old=model.nonlinear(old)
        f0=model.force(state.t)
        first,s1=self._stage(model,old+ETA*dt*f0,state.r,ETA*nonlinear_old,dt)
        rhs=first-model.nu*dt*(1-2*ETA)*(model.ops.K@first)
        rhs+=dt*(-f0+(1-DELTA)*model.force(state.t+ETA*dt))
        scalar_rhs=(1-self.gamma*dt*(1-2*ETA))*s1.r
        second,s2=self._stage(model,rhs,scalar_rhs,-nonlinear_old+(1-DELTA)*model.nonlinear(first),dt)
        return Trial(model.state(state.t+dt,second,s2.r),[s1,s2])

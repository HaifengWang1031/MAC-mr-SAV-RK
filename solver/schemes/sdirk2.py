"""Incremental form of the ordinary two-stage IMEX-SDIRK2 comparator."""
import numpy as np
from ..core import State, Stage, Trial
from ..mac_ns import MACNavierStokes

ETA=1-1/np.sqrt(2.)
DELTA=-1/np.sqrt(2.)

class SDIRK2:
    name='sdirk2'

    def step(self, model: MACNavierStokes, state: State, dt: float) -> Trial:
        if not np.isfinite(dt) or dt<=0: raise ValueError('Invalid step size')
        old=model.vector(state)
        nonlinear_old=model.nonlinear(old)
        f0=model.force(state.t)
        first=model.backend.solve(old+ETA*dt*(f0-nonlinear_old),mass=1.,viscosity=model.nu*ETA*dt)
        rhs=first.velocity-model.nu*dt*(1-2*ETA)*(model.ops.K@first.velocity)
        rhs+=dt*(-f0+(1-DELTA)*model.force(state.t+ETA*dt))
        rhs-=dt*(-nonlinear_old+(1-DELTA)*model.nonlinear(first.velocity))
        second=model.backend.solve(rhs,mass=1.,viscosity=model.nu*ETA*dt)
        stages=[Stage(s.pressure/dt,s.residual,s.divergence_inf) for s in (first,second)]
        return Trial(model.state(state.t+dt,second.velocity),stages)

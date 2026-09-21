"""Incremental form of the ordinary two-stage IMEX-SDIRK2 comparator.

Written against the model seam (`solver/model.py`): the stage algebra, the operator
access and the stage records all go through the model, so the same code drives any
discretisation that supplies the seam instead of being re-implemented per backend.
`docs/model-seam-spec.md` records the scope and the accepted rounding shift.
"""
from typing import Any
import numpy as np
from ..core import Trial
from ..model import Model

ETA=1-1/np.sqrt(2.)
DELTA=-1/np.sqrt(2.)

class SDIRK2:
    name='sdirk2'

    def step(self, model: Model, state: Any, dt: float) -> Trial:
        if not np.isfinite(dt) or dt<=0: raise ValueError('Invalid step size')
        old=model.vector(state)
        nonlinear_old=model.nonlinear(old)
        f0=model.force(state.t)
        viscosity=model.nu*ETA*dt
        first=model.solve(model.combine((1.,old),(ETA*dt,f0),(-ETA*dt,nonlinear_old)),
                          mass=1.,viscosity=viscosity)
        f1=model.force(state.t+ETA*dt)
        n1=model.nonlinear(first.velocity)
        rhs=model.combine((1.,first.velocity),(-model.nu*dt*(1-2*ETA),model.apply_K(first.velocity)),
                          (-dt,f0),(dt*(1-DELTA),f1),(dt,nonlinear_old),(-dt*(1-DELTA),n1))
        second=model.solve(rhs,mass=1.,viscosity=viscosity)
        # The stage pressure is an increment: divided by dt. Expressed as a scaled copy
        # so the same line works on a vector type that has no division.
        stages=[model.stage(model.combine((1./dt,s.pressure)),s.residual,s.divergence_inf)
                for s in (first,second)]
        return Trial(model.state(state.t+dt,second.velocity),stages)

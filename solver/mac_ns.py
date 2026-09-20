"""Velocity-pressure MAC model with an injectable coupled Stokes backend."""
from collections.abc import Callable
import numpy as np
from .core import State
from .mac.grid import MACGrid, Array
from .mac.operators import MACOperators
from .mac.stokes import DirectStokes, StokesBackend
from .mac.kernels import convection

class MACNavierStokes:
    def __init__(self, grid: MACGrid, nu: float, force: Callable[[float], Array] | None = None,
                 backend: StokesBackend | None = None, cache_size: int = 4) -> None:
        if not np.isfinite(nu) or nu<=0: raise ValueError('nu must be positive and finite')
        self.grid=grid
        self.nu=nu
        self.ops=MACOperators(grid)
        self.backend=backend if backend is not None else DirectStokes(self.ops,cache_size)
        self.force=force if force is not None else lambda t: np.zeros(grid.size)

    def nonlinear(self, velocity: Array) -> Array:
        u,v=self.grid.unpack(velocity)
        cu,cv=convection(u,v,self.grid.hx,self.grid.hy)
        return self.grid.pack(cu,cv)

    def vector(self, state: State) -> Array:
        return self.grid.pack(state.u,state.v)

    def state(self, t: float, velocity: Array, r: float = 0.) -> State:
        u,v=self.grid.unpack(velocity)
        return State(t,u,v,r)

    def diagnostics(self, state: State) -> dict[str,float]:
        velocity=self.vector(state)
        kinetic=.5*self.grid.inner(velocity,velocity)
        return {'kinetic':kinetic,'modified_energy':kinetic+.5*(state.r-1)**2,
                'divergence_inf':float(np.max(np.abs(self.ops.D@velocity))),
                'h1_seminorm_squared':self.grid.inner(velocity,self.ops.K@velocity),'r':state.r}

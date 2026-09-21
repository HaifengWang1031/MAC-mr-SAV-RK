"""Velocity-pressure MAC model with an injectable coupled Stokes backend."""
from collections.abc import Callable, Sequence
import numpy as np
from .core import State, Stage
from .mac.grid import MACGrid, Array
from .mac.operators import MACOperators
from .mac.stokes import DirectStokes, StokesBackend, StokesResult
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

    # --- the model seam (solver/model.py): behaviour-identical wrappers over the
    # operators and the backend, so the schemes no longer reach for grid/ops/backend.

    def combine(self, *terms: tuple[float, Array]) -> Array:
        """Sum of scalar*field pairs in the given order; sized by the first field."""
        total=np.zeros_like(terms[0][1],dtype=float)
        for coefficient,field in terms:
            total+=coefficient*field
        return total

    def apply_K(self, velocity: Array) -> Array:
        return self.ops.K@velocity

    def apply_G(self, pressure: Array) -> Array:
        return self.ops.G@np.asarray(pressure).reshape(-1)

    def apply_D(self, velocity: Array) -> Array:
        return self.ops.D@velocity

    def inner(self, left: Array, right: Array) -> float:
        return self.grid.inner(left,right)

    def max_abs(self, field: Array) -> float:
        return float(np.max(np.abs(field)))

    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> StokesResult:
        return self.backend.solve(rhs,mass=mass,viscosity=viscosity)

    def solve_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float) -> list[StokesResult]:
        """One batched solve, split into per-column results (splitting is exact)."""
        solved=self.backend.solve(np.column_stack(columns),mass=mass,viscosity=viscosity)
        return [StokesResult(solved.velocity[:,k],solved.pressure[:,:,k],solved.residual,solved.divergence_inf)
                for k in range(len(columns))]

    def stage(self, pressure: Array, residual: float, divergence_inf: float, r: float = 0.,
              candidates: Sequence[float] = (), root_residuals: Sequence[float] = (),
              scalar_residual: float = 0.) -> Stage:
        return Stage(pressure,residual,divergence_inf,r,list(candidates),list(root_residuals),scalar_residual)

    def diagnostics(self, state: State) -> dict[str,float]:
        velocity=self.vector(state)
        kinetic=.5*self.grid.inner(velocity,velocity)
        return {'kinetic':kinetic,'modified_energy':kinetic+.5*(state.r-1)**2,
                'divergence_inf':float(np.max(np.abs(self.ops.D@velocity))),
                'h1_seminorm_squared':self.grid.inner(velocity,self.ops.K@velocity),'r':state.r}

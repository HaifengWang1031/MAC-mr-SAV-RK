"""Distributed model and unchanged SDIRK stage equations with global SAV products."""
from dataclasses import dataclass,field
from typing import Any,Callable
from collections.abc import Sequence
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from ..mac.grid import MACGrid
from ..model import SolveResult
from .stokes import ParallelStokes


def combine(*terms: tuple[float,Any]) -> Any:
    result=terms[0][1].duplicate();result.set(0.)
    for coefficient,vector in terms:result.axpy(coefficient,vector)
    return result

@dataclass
class DistributedState:
    t: float
    velocity: Any
    r: float=0.

@dataclass
class DistributedStage:
    pressure: Any
    residual: float
    divergence_inf: float
    r: float=0.
    candidates: list[float]=field(default_factory=list)
    root_residuals: list[float]=field(default_factory=list)
    scalar_residual: float=0.

class ParallelNS:
    def __init__(self,grid: MACGrid,nu: float,*,comm: Any=MPI.COMM_WORLD,
                 tolerance: float=1e-10,max_iterations: int=300,cache_size: int=4) -> None:
        if not np.isfinite(nu) or nu<=0:raise ValueError('Positive finite nu required')
        self.grid,self.nu,self.comm=grid,nu,comm
        self.stokes=ParallelStokes(grid,comm=comm,tolerance=tolerance,max_iterations=max_iterations,cache_size=cache_size)
        self.force: Callable[[float],Any]=lambda t:self.zero()
        self.buffers: dict[str,Any]={}

    def zero(self) -> Any:
        result=self.stokes.layout.template.duplicate();result.set(0.);return result

    def initial(self,amplitude: float=.1) -> DistributedState:
        g=self.grid
        def psi(x: Any,y: Any) -> Any:
            value=amplitude*np.sin(np.pi*x/g.lx)**2*np.sin(np.pi*y/g.ly)**2
            return np.where((x<=0)|(x>=g.lx)|(y<=0)|(y>=g.ly),0.,value)
        velocity=self.stokes.layout.vector(lambda x,y:(psi(x,y+g.hy/2)-psi(x,y-g.hy/2))/g.hy,
                                          lambda x,y:-(psi(x+g.hx/2,y)-psi(x-g.hx/2,y))/g.hx)
        return DistributedState(0.,velocity)

    def viscous(self,velocity: Any) -> Any:
        result=velocity.duplicate();self.stokes.K.mult(velocity,result);return result

    def nonlinear(self,velocity: Any) -> Any:
        return self.stokes.layout.nonlinear(velocity)

    def inner(self,left: Any,right: Any) -> float:
        return float(self.grid.area*left.dot(right))

    def diagnostics(self,state: DistributedState) -> dict[str,float]:
        kinetic=.5*self.inner(state.velocity,state.velocity)
        lap=self.viscous(state.velocity)
        h1=self.inner(state.velocity,lap);lap.destroy()
        div=self.stokes.full_D.createVecLeft();self.stokes.full_D.mult(state.velocity,div)
        divergence=float(div.norm(PETSc.NormType.NORM_INFINITY));div.destroy()
        return {'kinetic':kinetic,'modified_energy':kinetic+.5*(state.r-1)**2,
                'h1_seminorm_squared':h1,'divergence_inf':divergence,'r':state.r}

    # --- the model seam (solver/model.py): the distributed realisation supplies the
    # same members as the serial one, so a single scheme implementation drives both.

    def vector(self,state: DistributedState) -> Any:
        return state.velocity

    def state(self,t: float,velocity: Any,r: float=0.) -> DistributedState:
        return DistributedState(t,velocity,r)

    def combine(self,*terms: tuple[float,Any]) -> Any:
        return combine(*terms)

    def _buffer(self,name: str,create: Callable[[],Any]) -> Any:
        """A vector the model owns and reuses, because the schemes cannot destroy one."""
        result=self.buffers.get(name)
        if result is None:
            result=create()
            self.buffers[name]=result
        return result

    def apply_K(self,velocity: Any) -> Any:
        """K v in a reused buffer, valid until the next apply_K call."""
        result=self._buffer('K',self.stokes.layout.template.duplicate)
        self.stokes.K.mult(velocity,result)
        return result

    def apply_G(self,pressure: Any) -> Any:
        """G p in a reused buffer, valid until the next apply_G call."""
        result=self._buffer('G',self.stokes.layout.template.duplicate)
        self.stokes.G.mult(pressure,result)
        return result

    def apply_D(self,velocity: Any) -> Any:
        """D v on all rows, matching the serial operator: the solver's system drops one row,
        this one does not, so the buffer takes the full pressure layout from the matrix."""
        result=self._buffer('D',self.stokes.full_D.createVecLeft)
        self.stokes.full_D.mult(velocity,result)
        return result

    def max_abs(self,field: Any) -> float:
        return float(field.norm(PETSc.NormType.NORM_INFINITY))

    def solve(self,rhs: Any,*,mass: float,viscosity: float) -> Any:
        return self.stokes.solve(rhs,mass=mass,viscosity=viscosity)

    def solve_columns(self,columns: Sequence[Any],*,mass: float,viscosity: float) -> list[SolveResult]:
        """One solve per right-hand side: this backend cannot batch them."""
        return [self.stokes.solve(column,mass=mass,viscosity=viscosity) for column in columns]

    def stage(self,pressure: Any,residual: float,divergence_inf: float,r: float=0.,
              candidates: Sequence[float]=(),root_residuals: Sequence[float]=(),
              scalar_residual: float=0.) -> DistributedStage:
        return DistributedStage(pressure,residual,divergence_inf,r,list(candidates),
                                list(root_residuals),scalar_residual)

    def close(self) -> None:
        for buffer in self.buffers.values():buffer.destroy()
        self.buffers.clear()
        self.stokes.close()

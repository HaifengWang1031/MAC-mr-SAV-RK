"""Distributed model and unchanged SDIRK stage equations with global SAV products."""
from dataclasses import dataclass,field
from typing import Any,Callable
from collections.abc import Sequence
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from ..mac.grid import MACGrid
from ..model import SolveResult
from ..schemes.sdirk2 import ETA,DELTA
from ..schemes.roots import real_roots
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

@dataclass
class DistributedTrial:
    state: DistributedState
    stages: list[DistributedStage]

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

class ParallelSDIRK2:
    def __init__(self,*,sav: bool=True,gamma: float=1.) -> None:
        if not np.isfinite(gamma) or gamma<0:raise ValueError('Invalid gamma')
        self.sav,self.gamma=sav,gamma

    def _stage(self,model: ParallelNS,rhs: Any,scalar_rhs: float,B: Any,dt: float) -> tuple[Any,DistributedStage]:
        if not self.sav:
            combined=combine((1.,rhs),(-dt,B))
            sol=model.stokes.solve(combined,mass=1.,viscosity=model.nu*ETA*dt);combined.destroy()
            sol.pressure.scale(1/dt)
            return sol.velocity,DistributedStage(sol.pressure,sol.residual,sol.divergence_inf)
        first=model.stokes.solve(rhs,mass=1.,viscosity=model.nu*ETA*dt)
        second=model.stokes.solve(B,mass=1.,viscosity=model.nu*ETA*dt)
        alpha=model.inner(B,first.velocity);beta=model.inner(B,second.velocity)
        if beta < -1e-12*(1+np.sqrt(model.inner(B,B)*model.inner(second.velocity,second.velocity))):
            raise RuntimeError('Negative Stokes response energy')
        beta=max(beta,0.)
        message: Any=None
        if model.comm.rank==0:
            try:
                choice=real_roots(np.array([dt*dt*beta,dt*dt*beta,1+self.gamma*ETA*dt+dt*alpha-dt*dt*beta,
                                           -scalar_rhs+dt*alpha-dt*dt*beta]))
                message=(None,choice.selected,choice.candidates,choice.residuals)
            except Exception as exc:message=(str(exc),0.,[],[])
        error,r,candidates,root_residuals=model.comm.bcast(message,root=0)
        if error:raise RuntimeError(error)
        velocity=combine((1.,first.velocity),(-dt*(1-r*r),second.velocity))
        pressure=combine((1/dt,first.pressure),(-(1-r*r),second.pressure))
        scalar=(1+self.gamma*ETA*dt)*r-scalar_rhs+dt*(1+r)*model.inner(B,velocity)
        scalar_residual=abs(scalar)/(1+abs(scalar_rhs)+abs(r))
        lap=model.viscous(velocity);gradient=model.stokes.gradient(pressure)
        defect=combine((1.,velocity),(model.nu*ETA*dt,lap),(dt,gradient),(-1.,rhs),(dt*(1-r*r),B))
        norm_inf=PETSc.NormType.NORM_INFINITY
        residual=defect.norm(norm_inf)/(1+rhs.norm(norm_inf)+dt*abs(1-r*r)*B.norm(norm_inf))
        div=model.stokes.full_D.createVecLeft();model.stokes.full_D.mult(velocity,div)
        divergence=div.norm(norm_inf)
        for vec in (first.velocity,second.velocity,first.pressure,second.pressure,lap,gradient,defect,div):vec.destroy()
        if not np.isfinite([residual,scalar_residual,divergence]).all() or max(residual,scalar_residual,divergence)>1e-8:
            velocity.destroy();pressure.destroy();raise RuntimeError(f'Stage residuals {residual}, {scalar_residual}, {divergence}')
        return velocity,DistributedStage(pressure,float(residual),float(divergence),r,candidates,root_residuals,float(scalar_residual))

    def step(self,model: ParallelNS,state: DistributedState,dt: float) -> DistributedTrial:
        if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid step size')
        n0=model.nonlinear(state.velocity);f0=model.force(state.t)
        rhs1=combine((1.,state.velocity),(ETA*dt,f0));b1=combine((ETA,n0))
        first,s1=self._stage(model,rhs1,state.r,b1,dt)
        k1=model.viscous(first);n1=model.nonlinear(first);f1=model.force(state.t+ETA*dt)
        rhs2=combine((1.,first),(-model.nu*dt*(1-2*ETA),k1),(-dt,f0),(dt*(1-DELTA),f1))
        b2=combine((-1.,n0),(1-DELTA,n1))
        second,s2=self._stage(model,rhs2,(1-self.gamma*dt*(1-2*ETA))*s1.r,b2,dt)
        for vec in (n0,f0,rhs1,b1,first,k1,n1,f1,rhs2,b2):vec.destroy()
        return DistributedTrial(DistributedState(state.t+dt,second,s2.r),[s1,s2])

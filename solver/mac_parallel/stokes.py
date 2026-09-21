"""Distributed shifted Stokes: FGMRES and a block triangular AMG preconditioner."""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from .layout import SlabLayout
from .failures import local_call
from ..mac.grid import MACGrid

@dataclass
class ParallelSolution:
    velocity: Any
    pressure: Any
    residual: float
    divergence_inf: float
    iterations: int

class BlockPreconditioner:
    """Approximate lower inverse; Schur inverse ~ mass*Poisson^-1 + viscosity*I."""
    def __init__(self, L: Any, Q: Any, D: Any, mass: float, viscosity: float) -> None:
        self.mass,self.viscosity,self.D=mass,viscosity,D
        # Inner solves are tunable through PETSC_OPTIONS with the vel_/pres_ prefixes;
        # defaults are unchanged when no options are supplied.
        self.velocity=PETSc.KSP().create(comm=L.comm)
        self.velocity.setOptionsPrefix('vel_')
        self.velocity.setOperators(L);self.velocity.setType('preonly');self.velocity.getPC().setType('gamg' if viscosity>0 else 'jacobi')
        self.velocity.setFromOptions();self.velocity.setUp()
        self.pressure=PETSc.KSP().create(comm=Q.comm)
        self.pressure.setOptionsPrefix('pres_')
        self.pressure.setOperators(Q);self.pressure.setType('preonly');self.pressure.getPC().setType('gamg')
        self.pressure.setFromOptions();self.pressure.setUp()
        self.work=Q.createVecRight();self.poisson=Q.createVecRight()
        self.velocity_seconds=0.;self.pressure_seconds=0.

    def apply(self, pc: Any, rhs: Any, out: Any) -> None:
        ru,rp=rhs.getNestSubVecs();zu,zp=out.getNestSubVecs()
        start=perf_counter()
        self.velocity.solve(ru,zu)
        self.velocity_seconds+=perf_counter()-start
        self.D.mult(zu,self.work);self.work.axpy(1.,rp)
        start=perf_counter()
        self.pressure.solve(self.work,self.poisson)
        self.pressure_seconds+=perf_counter()-start
        self.poisson.copy(zp);zp.scale(-self.mass);zp.axpy(-self.viscosity,self.work)

    def close(self) -> None:
        self.velocity.destroy();self.pressure.destroy();self.work.destroy();self.poisson.destroy()

class ParallelStokes:
    def __init__(self, grid: MACGrid, *, comm: Any = MPI.COMM_WORLD,
                 tolerance: float = 1e-10, max_iterations: int = 300, cache_size: int = 4,
                 refinement_attempts: int = 3, refinement_factor: float = 0.01,
                 tolerance_floor: float = 1e-14) -> None:
        if tolerance<=0 or not np.isfinite(tolerance) or max_iterations<1 or cache_size<1:
            raise ValueError('Invalid iteration controls')
        if refinement_attempts<1 or not 0<refinement_factor<1 or tolerance_floor<=0:
            raise ValueError('Invalid divergence-refinement controls')
        self.grid,self.comm,self.tolerance=grid,comm,tolerance
        self.max_iterations,self.cache_size=max_iterations,cache_size
        self.refinement_attempts,self.refinement_factor=refinement_attempts,refinement_factor
        self.tolerance_floor=tolerance_floor
        self.layout=SlabLayout(grid,comm)
        self.cache: OrderedDict = OrderedDict()
        self._retired_velocity_seconds=0.
        self._retired_pressure_seconds=0.
        self.setup_seconds=0.;self.solve_seconds=0.;self.iterations: list[int]=[]
        self.attempts: list[int]=[]
        try:
            self.K,self.D,self.full_D,self.Q=self._assemble()
        except Exception:
            self.layout.close()
            raise
        self.G=self.D.copy();self.G.transpose();self.G.scale(-1.)

    def _assemble(self) -> tuple[Any,Any,Any,Any]:
        g=self.grid;l=self.layout
        def matrix(n: int,N: int,m: int,M: int) -> Any:
            return PETSc.Mat().createAIJ(size=((n,N),(m,M)),nnz=5,comm=self.comm)
        K=matrix(l.local_n,g.size,l.local_n,g.size)
        D=matrix(l.local_p,g.np-1,l.local_n,g.size)
        full=matrix(l.rows*g.nx,g.np,l.local_n,g.size)
        Q=matrix(l.local_p,g.np-1,l.local_p,g.np-1)
        def fill_local_rows() -> None:
            for j in range(l.start,l.end):
                for i in range(1,g.nx):
                    row=l.u_index(j,i); diagonal=2/g.hx**2+(3 if j in (0,g.ny-1) else 2)/g.hy**2
                    cols=[row];vals=[diagonal]
                    for jj,ii,value in ((j,i-1,-1/g.hx**2),(j,i+1,-1/g.hx**2),(j-1,i,-1/g.hy**2),(j+1,i,-1/g.hy**2)):
                        if 0<=jj<g.ny and 0<ii<g.nx: cols.append(l.u_index(jj,ii));vals.append(value)
                    K.setValues(row,cols,vals)
            for j in range(l.start+1,min(l.end,g.ny-1)+1):
                for i in range(g.nx):
                    row=l.v_index(j,i);cols=[row];vals=[(3 if i in (0,g.nx-1) else 2)/g.hx**2+2/g.hy**2]
                    for jj,ii,value in ((j,i-1,-1/g.hx**2),(j,i+1,-1/g.hx**2),(j-1,i,-1/g.hy**2),(j+1,i,-1/g.hy**2)):
                        if 0<jj<g.ny and 0<=ii<g.nx: cols.append(l.v_index(jj,ii));vals.append(value)
                    K.setValues(row,cols,vals)
            for j in range(l.start,l.end):
                for i in range(g.nx):
                    row=j*g.nx+i;cols=[];vals=[]
                    if i<g.nx-1:cols.append(l.u_index(j,i+1));vals.append(1/g.hx)
                    if i>0:cols.append(l.u_index(j,i));vals.append(-1/g.hx)
                    if j<g.ny-1:cols.append(l.v_index(j+1,i));vals.append(1/g.hy)
                    if j>0:cols.append(l.v_index(j,i));vals.append(-1/g.hy)
                    full.setValues(row,cols,vals)
                    if row==g.np-1:continue
                    D.setValues(row,cols,vals)
                    qcols=[];qvals=[];diagonal=0.
                    for jj,ii,value in ((j,i-1,1/g.hx**2),(j,i+1,1/g.hx**2),(j-1,i,1/g.hy**2),(j+1,i,1/g.hy**2)):
                        if 0<=jj<g.ny and 0<=ii<g.nx:
                            diagonal+=value
                            if jj*g.nx+ii<g.np-1:qcols.append(jj*g.nx+ii);qvals.append(-value)
                    Q.setValues(row,[row]+qcols,[diagonal]+qvals)
        try:local_call(self.comm,'matrix row assembly',fill_local_rows)
        except Exception:
            for mat in (K,D,full,Q):mat.destroy()
            raise
        for mat in (K,D,full,Q):mat.assemble()
        return K,D,full,Q

    def pressure_vector(self,function: Callable) -> Any:
        l=self.layout;g=self.grid
        x,y=np.meshgrid((np.arange(g.nx)+.5)*g.hx,(np.arange(l.start,l.end)+.5)*g.hy)
        values=np.broadcast_to(function(x,y),x.shape).ravel()[:l.local_p]
        result=l.p_template.duplicate()
        result.array[:]=values-function((g.nx-.5)*g.hx,(g.ny-.5)*g.hy)
        return result

    def gradient(self,pressure: Any) -> Any:
        result=self.layout.template.duplicate();self.G.mult(pressure,result);return result

    def _system(self,mass: float,viscosity: float) -> tuple[Any,...]:
        key=(float(mass),float(viscosity))
        if key not in self.cache:
            start=perf_counter()
            L=self.K.copy();L.scale(viscosity);L.shift(mass)
            lower=self.D.copy();lower.scale(-1.)
            blocks: Any=[[L,self.G],[lower,None]]
            A=PETSc.Mat().createNest(blocks,comm=self.comm);A.assemble();A.setNestVecType('nest')
            pc=BlockPreconditioner(L,self.Q,self.D,mass,viscosity)
            ksp=PETSc.KSP().create(comm=self.comm);ksp.setOperators(A)
            ksp.setType('fgmres');ksp.setGMRESRestart(60)
            ksp.setTolerances(rtol=self.tolerance,atol=self.tolerance*1e-3,max_it=self.max_iterations)
            ksp.getPC().setType('python');ksp.getPC().setPythonContext(pc)
            ksp.setFromOptions()
            ksp.setUp()
            self.cache[key]=(L,lower,A,pc,ksp)
            self.setup_seconds+=perf_counter()-start
            if len(self.cache)>self.cache_size:self._retire(self.cache.popitem(last=False)[1])
        self.cache.move_to_end(key)
        return self.cache[key]

    def solve(self,rhs: Any,*,mass: float,viscosity: float) -> ParallelSolution:
        if not np.isfinite([mass,viscosity]).all() or min(mass,viscosity)<0 or mass+viscosity<=0:
            raise ValueError('Invalid shifted Stokes coefficients')
        L,_,_,_,ksp=self._system(mass,viscosity)
        u=self.layout.template.duplicate();p=self.layout.p_template.duplicate();zero=p.duplicate();zero.set(0)
        b=PETSc.Vec().createNest([rhs,zero],comm=self.comm)
        solution=PETSc.Vec().createNest([u,p],comm=self.comm)
        defect: Any=u.duplicate();gradient=u.duplicate();div=self.full_D.createVecLeft()
        # The nested KSP residual is a mixed norm in which the continuity rows (coefficients
        # ~1/h) are far smaller than the momentum rows (~1/h^2), so a converged momentum
        # residual does not by itself bound the divergence. Momentum and divergence are both
        # required here: if only the momentum part is satisfied the tolerance is tightened and
        # the solve is restarted from the current iterate, so the divergence participates in
        # the stopping decision instead of only failing the gate afterwards.
        gate=100*self.tolerance*(1+rhs.norm())
        reason=0;iterations=0;attempts=0;rtol=self.tolerance
        residual=divergence=float('inf')
        while True:
            attempts+=1
            ksp.setTolerances(rtol=rtol,atol=rtol*1e-3,max_it=self.max_iterations)
            start=perf_counter();ksp.solve(b,solution);self.solve_seconds+=perf_counter()-start
            iterations+=ksp.getIterationNumber();reason=ksp.getConvergedReason()
            L.mult(u,defect);self.G.mult(p,gradient);defect.axpy(1.,gradient);defect.axpy(-1.,rhs)
            residual=defect.norm(PETSc.NormType.NORM_INFINITY)/(1+rhs.norm(PETSc.NormType.NORM_INFINITY))
            self.full_D.mult(u,div);divergence=div.norm(PETSc.NormType.NORM_INFINITY)
            if reason<=0 or max(residual,divergence)<=gate or attempts>=self.refinement_attempts:
                break
            rtol=max(rtol*self.refinement_factor,self.tolerance_floor)
            ksp.setInitialGuessNonzero(True)
        for vec in (b,solution,zero,defect,gradient,div):vec.destroy()
        self.iterations.append(iterations);self.attempts.append(attempts)
        if reason<=0 or not np.isfinite([residual,divergence]).all() or max(residual,divergence)>gate:
            u.destroy();p.destroy()
            raise RuntimeError(f'FGMRES reason={reason}, residual={residual}, divergence={divergence}, '
                               f'iterations={iterations}, attempts={attempts}')
        return ParallelSolution(u,p,float(residual),float(divergence),iterations)

    @staticmethod
    def _destroy(system: tuple[Any,...]) -> None:
        L,lower,A,pc,ksp=system;ksp.destroy();pc.close();A.destroy();lower.destroy();L.destroy()

    def _retire(self, system: tuple[Any,...]) -> None:
        """Retain application times before releasing an evicted or closed system."""
        self._retired_velocity_seconds+=system[3].velocity_seconds
        self._retired_pressure_seconds+=system[3].pressure_seconds
        self._destroy(system)

    @property
    def preconditioner_seconds(self) -> tuple[float,float]:
        """(velocity block, pressure block) application time over all solves, including retired systems."""
        return (self._retired_velocity_seconds+sum(system[3].velocity_seconds for system in self.cache.values()),
                self._retired_pressure_seconds+sum(system[3].pressure_seconds for system in self.cache.values()))

    def close(self) -> None:
        for system in self.cache.values():self._retire(system)
        self.cache.clear()
        for mat in (self.G,self.D,self.full_D,self.K,self.Q):mat.destroy()
        self.layout.close()

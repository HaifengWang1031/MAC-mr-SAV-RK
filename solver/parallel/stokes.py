"""Distributed shifted Stokes: FGMRES and a block triangular AMG preconditioner."""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from .layout import SlabLayout
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
                 tolerance: float = 1e-10, max_iterations: int = 300, cache_size: int = 4) -> None:
        if tolerance<=0 or not np.isfinite(tolerance) or max_iterations<1 or cache_size<1:
            raise ValueError('Invalid iteration controls')
        self.grid,self.comm,self.tolerance=grid,comm,tolerance
        self.max_iterations,self.cache_size=max_iterations,cache_size
        self.layout=SlabLayout(grid,comm)
        self.cache: OrderedDict = OrderedDict()
        self.setup_seconds=0.;self.solve_seconds=0.;self.iterations: list[int]=[]
        self.K,self.D,self.full_D,self.Q=self._assemble()
        self.G=self.D.copy();self.G.transpose();self.G.scale(-1.)

    def _assemble(self) -> tuple[Any,Any,Any,Any]:
        g=self.grid;l=self.layout
        def matrix(n: int,N: int,m: int,M: int) -> Any:
            return PETSc.Mat().createAIJ(size=((n,N),(m,M)),nnz=5,comm=self.comm)
        K=matrix(l.local_n,g.size,l.local_n,g.size)
        D=matrix(l.local_p,g.np-1,l.local_n,g.size)
        full=matrix(l.rows*g.nx,g.np,l.local_n,g.size)
        Q=matrix(l.local_p,g.np-1,l.local_p,g.np-1)
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
            if len(self.cache)>self.cache_size:self._destroy(self.cache.popitem(last=False)[1])
        self.cache.move_to_end(key)
        return self.cache[key]

    def solve(self,rhs: Any,*,mass: float,viscosity: float) -> ParallelSolution:
        if not np.isfinite([mass,viscosity]).all() or min(mass,viscosity)<0 or mass+viscosity<=0:
            raise ValueError('Invalid shifted Stokes coefficients')
        L,_,_,_,ksp=self._system(mass,viscosity)
        u=self.layout.template.duplicate();p=self.layout.p_template.duplicate();zero=p.duplicate();zero.set(0)
        b=PETSc.Vec().createNest([rhs,zero],comm=self.comm)
        solution=PETSc.Vec().createNest([u,p],comm=self.comm)
        start=perf_counter();ksp.solve(b,solution);self.solve_seconds+=perf_counter()-start
        iterations=ksp.getIterationNumber();reason=ksp.getConvergedReason()
        defect: Any=u.duplicate();L.mult(u,defect);gradient=self.gradient(p);defect.axpy(1.,gradient);defect.axpy(-1.,rhs)
        residual=defect.norm(PETSc.NormType.NORM_INFINITY)/(1+rhs.norm(PETSc.NormType.NORM_INFINITY))
        div=self.full_D.createVecLeft();self.full_D.mult(u,div);divergence=div.norm(PETSc.NormType.NORM_INFINITY)
        for vec in (b,solution,zero,defect,gradient,div):vec.destroy()
        self.iterations.append(iterations)
        if reason<=0 or not np.isfinite([residual,divergence]).all() or max(residual,divergence)>100*self.tolerance*(1+rhs.norm()):
            u.destroy();p.destroy();raise RuntimeError(f'FGMRES reason={reason}, residual={residual}, divergence={divergence}, iterations={iterations}')
        return ParallelSolution(u,p,float(residual),float(divergence),iterations)

    @staticmethod
    def _destroy(system: tuple[Any,...]) -> None:
        L,lower,A,pc,ksp=system;ksp.destroy();pc.close();A.destroy();lower.destroy();L.destroy()

    @property
    def preconditioner_seconds(self) -> tuple[float,float]:
        """(velocity block, pressure block) application time, summed over cached systems."""
        return (sum(system[3].velocity_seconds for system in self.cache.values()),
                sum(system[3].pressure_seconds for system in self.cache.values()))

    def close(self) -> None:
        for system in self.cache.values():self._destroy(system)
        self.cache.clear()
        for mat in (self.G,self.D,self.full_D,self.K,self.Q):mat.destroy()
        self.layout.close()

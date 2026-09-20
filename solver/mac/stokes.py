"""Coupled shifted Stokes solve, exact pressure gauge and bounded LU reuse."""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from .grid import Array
from .operators import MACOperators

@dataclass
class StokesResult:
    velocity: Array
    pressure: Array
    residual: float
    divergence_inf: float

class StokesBackend(Protocol):
    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> StokesResult: ...

class DirectStokes:
    def __init__(self, operators: MACOperators, cache_size: int = 4, tolerance: float = 1e-9) -> None:
        if cache_size < 1 or tolerance <= 0 or not np.isfinite(tolerance): raise ValueError('Invalid solver controls')
        self.ops=operators
        self.cache_size=cache_size
        self.tolerance=tolerance
        self.cache: OrderedDict = OrderedDict()
        self.factorizations=0
        self.factorization_seconds=0.0
        self.solve_seconds=0.0

    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> StokesResult:
        if not np.isfinite([mass,viscosity]).all() or min(mass,viscosity)<0 or mass+viscosity<=0:
            raise ValueError('Require mass,viscosity >= 0 and nonzero total')
        if rhs.ndim not in (1,2) or rhs.shape[0]!=self.ops.grid.size or not np.isfinite(rhs).all():
            raise ValueError('Invalid Stokes RHS')
        key=(float(mass),float(viscosity))
        if key not in self.cache:
            start=perf_counter()
            L=mass*sp.eye(self.ops.grid.size,format='csr')+viscosity*self.ops.K
            # One dependent continuity row and one pressure gauge are removed together.
            # All continuity rows are checked after the solve; pressure is then mean-zero.
            A=sp.bmat([[L,self.ops.G[:,:-1]],[-self.ops.D[:-1,:],None]],format='csc')
            self.cache[key]=(splu(A),L)
            self.factorizations+=1
            self.factorization_seconds+=perf_counter()-start
            if len(self.cache)>self.cache_size: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        lu,L=self.cache[key]
        two_dimensional=rhs.ndim==2
        force=rhs if two_dimensional else rhs[:,None]
        start=perf_counter()
        sol=lu.solve(np.vstack([force,np.zeros((self.ops.grid.np-1,force.shape[1]))]))
        self.solve_seconds+=perf_counter()-start
        velocity=sol[:self.ops.grid.size]
        pressure=np.vstack([sol[self.ops.grid.size:],np.zeros((1,force.shape[1]))])
        pressure-=pressure.mean(axis=0)
        defect=L@velocity+self.ops.G@pressure-force
        residual=float(np.max(np.abs(defect))/(1+np.max(np.abs(force))))
        divergence=float(np.max(np.abs(self.ops.D@velocity)))
        if not np.isfinite(sol).all() or residual>self.tolerance or divergence>self.tolerance*(1+np.max(np.abs(velocity))):
            raise RuntimeError(f'Stokes residual={residual:.3e}, divergence={divergence:.3e}')
        if not two_dimensional:
            return StokesResult(velocity[:,0],pressure[:,0].reshape(self.ops.grid.ny,self.ops.grid.nx),residual,divergence)
        return StokesResult(velocity,pressure.reshape(self.ops.grid.ny,self.ops.grid.nx,-1),residual,divergence)

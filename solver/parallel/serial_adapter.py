"""Single-rank PETSc adapter for the existing NumPy StokesBackend interface."""
import numpy as np
from mpi4py import MPI
from ..mac.grid import Array
from ..mac.operators import MACOperators
from ..mac.stokes import StokesResult
from .stokes import ParallelStokes

class PETScStokes:
    def __init__(self, operators: MACOperators, tolerance: float = 1e-10,
                 max_iterations: int = 300, cache_size: int = 4) -> None:
        self.backend=ParallelStokes(operators.grid,comm=MPI.COMM_SELF,tolerance=tolerance,
                                    max_iterations=max_iterations,cache_size=cache_size)

    def solve(self,rhs: Array,*,mass: float,viscosity: float) -> StokesResult:
        if rhs.ndim not in (1,2) or rhs.shape[0]!=self.backend.grid.size or not np.isfinite(rhs).all():
            raise ValueError('Invalid Stokes RHS')
        values=rhs[:,None] if rhs.ndim==1 else rhs
        velocities=[];pressures=[];residual=0.;divergence=0.
        for column in values.T:
            vector=self.backend.layout.template.duplicate();vector.array[:]=column
            result=self.backend.solve(vector,mass=mass,viscosity=viscosity)
            velocities.append(result.velocity.array.copy())
            pressures.append(self.backend.layout.gather_pressure(result.pressure))
            residual=max(residual,result.residual);divergence=max(divergence,result.divergence_inf)
            vector.destroy();result.velocity.destroy();result.pressure.destroy()
        u=np.column_stack(velocities);p=np.stack(pressures,axis=-1)
        if rhs.ndim==1:return StokesResult(u[:,0],p[:,:,0],residual,divergence)
        return StokesResult(u,p,residual,divergence)

    def close(self) -> None:
        self.backend.close()

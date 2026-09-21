"""Y-slab ownership with rank-contiguous MAC velocities and one-row halos."""
from typing import Any, Callable
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from ..mac.grid import MACGrid
from ..mac.kernels import convection
from .failures import local_call

class SlabLayout:
    def __init__(self, grid: MACGrid, comm: Any = MPI.COMM_WORLD) -> None:
        self.grid,self.comm=grid,comm
        self.rank,self.size=comm.rank,comm.size
        if comm.size>grid.ny: raise ValueError('MPI ranks must not exceed ny')
        self.edges=np.linspace(0,grid.ny,comm.size+1,dtype=int)
        self.owners=np.searchsorted(self.edges,np.arange(grid.ny),side='right')-1
        self.start,self.end=map(int,self.edges[comm.rank:comm.rank+2])
        self.rows=self.end-self.start
        self.vrows=min(self.end,grid.ny-1)-self.start
        self.local_u=self.rows*(grid.nx-1)
        self.local_n=self.local_u+self.vrows*grid.nx
        self.offset=self.start*(grid.nx-1)+min(self.start,grid.ny-1)*grid.nx
        self.local_p=self.rows*grid.nx-(self.end==grid.ny)
        self.template=PETSc.Vec().createMPI((self.local_n,grid.size),comm=comm)
        self.p_template=PETSc.Vec().createMPI((self.local_p,grid.np-1),comm=comm)
        ids=[]; positions=[]
        ushape=(self.rows+2,grid.nx+1); vshape=(self.rows+3,grid.nx)
        for j in range(self.start-1,self.end+1):
            if 0<=j<grid.ny:
                for i in range(1,grid.nx):
                    ids.append(self.u_index(j,i)); positions.append((j-self.start+1)*(grid.nx+1)+i)
        for j in range(self.start-1,self.end+2):
            if 0<j<grid.ny:
                for i in range(grid.nx):
                    ids.append(self.v_index(j,i));positions.append(int(np.prod(ushape))+(j-self.start+1)*grid.nx+i)
        self.halo_positions=np.asarray(positions,dtype=int)
        self.halo_shapes=ushape,vshape
        self.halo=PETSc.Vec().createSeq(len(ids),comm=PETSc.COMM_SELF)
        source=PETSc.IS().createGeneral(ids,comm=comm)
        target=PETSc.IS().createStride(len(ids),comm=PETSc.COMM_SELF)
        self.scatter: Any=PETSc.Scatter().create(self.template,source,self.halo,target)
        source.destroy();target.destroy()

    def u_index(self,j: int,i: int) -> int:
        g=self.grid; rank=int(self.owners[j]); first=int(self.edges[rank])
        offset=first*(g.nx-1)+min(first,g.ny-1)*g.nx
        return offset+(j-first)*(g.nx-1)+i-1

    def v_index(self,j: int,i: int) -> int:
        g=self.grid; rank=int(self.owners[j-1]);first=int(self.edges[rank]);last=int(self.edges[rank+1])
        offset=first*(g.nx-1)+min(first,g.ny-1)*g.nx
        return offset+(last-first)*(g.nx-1)+(j-first-1)*g.nx+i

    def vector(self, u: Callable, v: Callable) -> Any:
        def evaluate() -> Any:
            g=self.grid
            xu,yu=np.meshgrid(np.arange(1,g.nx)*g.hx,(np.arange(self.start,self.end)+.5)*g.hy)
            xv,yv=np.meshgrid((np.arange(g.nx)+.5)*g.hx,np.arange(self.start+1,self.start+self.vrows+1)*g.hy)
            return np.concatenate([np.broadcast_to(u(xu,yu),xu.shape).ravel(),
                                   np.broadcast_to(v(xv,yv),xv.shape).ravel()])
        values=local_call(self.comm,'field evaluation',evaluate)
        result=self.template.duplicate()
        try:
            local_call(self.comm,'field assignment',lambda: np.copyto(result.array,values))
        except Exception:
            result.destroy()
            raise
        return result

    def nonlinear(self, velocity: Any) -> Any:
        self.scatter.scatter(velocity,self.halo,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        def evaluate() -> Any:
            us,vs=self.halo_shapes
            values=np.zeros(np.prod(us)+np.prod(vs))
            values[self.halo_positions]=self.halo.array
            u=values[:np.prod(us)].reshape(us);v=values[np.prod(us):].reshape(vs)
            cu,cv=convection(u,v,self.grid.hx,self.grid.hy)
            return np.concatenate([cu[1:self.rows+1,1:-1].ravel(),cv[2:self.vrows+2,:].ravel()])
        values=local_call(self.comm,'convection kernel',evaluate)
        result=self.template.duplicate()
        try:
            local_call(self.comm,'convection assignment',lambda: np.copyto(result.array,values))
        except Exception:
            result.destroy()
            raise
        return result

    def gather(self, velocity: Any, root: int = 0) -> Any:
        """Collect global fields only for output or explicit reference tests."""
        values=local_call(self.comm,'gather velocity preparation',lambda: velocity.array.copy())
        pieces=self.comm.gather(values,root=root)
        def assemble() -> Any:
            if self.rank!=root:return None
            g=self.grid;u=np.zeros((g.ny,g.nx+1));v=np.zeros((g.ny+1,g.nx))
            for rank,part in enumerate(pieces):
                first,last=map(int,self.edges[rank:rank+2]);n=(last-first)*(g.nx-1)
                u[first:last,1:-1]=part[:n].reshape(last-first,g.nx-1)
                v[first+1:min(last,g.ny-1)+1,:]=part[n:].reshape(min(last,g.ny-1)-first,g.nx)
            return u,v
        return local_call(self.comm,'gather velocity assembly',assemble)

    def gather_pressure(self, pressure: Any, root: int = 0) -> Any:
        values=local_call(self.comm,'gather pressure preparation',lambda: pressure.array.copy())
        pieces=self.comm.gather(values,root=root)
        def assemble() -> Any:
            if self.rank!=root:return None
            p=np.r_[np.concatenate(pieces),0.];p-=p.mean()
            return p.reshape(self.grid.ny,self.grid.nx)
        return local_call(self.comm,'gather pressure assembly',assemble)

    def close(self) -> None:
        self.scatter.destroy(); self.halo.destroy();self.template.destroy();self.p_template.destroy()

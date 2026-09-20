"""Compatible MAC matrices after eliminating zero normal boundary values."""
import numpy as np
import scipy.sparse as sp
from .grid import MACGrid

def _laplacian_1d(n: int, spacing: float, half_cell_wall: bool):
    diagonal = np.full(n,2.0)
    if half_cell_wall: diagonal[[0,-1]] = 3.0
    return sp.diags([-np.ones(n-1),diagonal,-np.ones(n-1)],[-1,0,1],format='csr')/spacing**2

class MACOperators:
    def __init__(self, grid: MACGrid) -> None:
        self.grid = grid
        nx,ny=grid.nx,grid.ny
        ku = sp.kron(sp.eye(ny),_laplacian_1d(nx-1,grid.hx,False)) + sp.kron(_laplacian_1d(ny,grid.hy,True),sp.eye(nx-1))
        kv = sp.kron(sp.eye(ny-1),_laplacian_1d(nx,grid.hx,True)) + sp.kron(_laplacian_1d(ny-1,grid.hy,False),sp.eye(nx))
        self.K = sp.block_diag([ku,kv],format='csr')
        rows,cols,values=[],[],[]
        for j in range(ny):
            for i in range(nx):
                entries=[]
                if i<nx-1: entries.append((j*(nx-1)+i,1/grid.hx))
                if i>0: entries.append((j*(nx-1)+i-1,-1/grid.hx))
                if j<ny-1: entries.append((grid.nu+j*nx+i,1/grid.hy))
                if j>0: entries.append((grid.nu+(j-1)*nx+i,-1/grid.hy))
                for col,value in entries:
                    rows.append(j*nx+i);cols.append(col);values.append(value)
        self.D=sp.csr_matrix((values,(rows,cols)),shape=(grid.np,grid.size))
        self.G=-self.D.T.tocsr()

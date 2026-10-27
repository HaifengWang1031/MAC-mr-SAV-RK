"""Compatible second-order MAC operators; one independent x-periodic face."""
import numpy as np
import scipy.sparse as sp

from ..mac.operators import _laplacian_1d
from .grid import ChannelGrid


class ChannelOperators:
    def __init__(self, grid: ChannelGrid) -> None:
        self.grid = grid
        nx, ny = grid.nx, grid.ny
        rows = np.arange(nx)
        forward = sp.csr_matrix((np.ones(nx), (rows, (rows+1) % nx)), shape=(nx, nx))
        dx = (forward-sp.eye(nx, format='csr'))/grid.hx
        kx = dx.T@dx
        ku = sp.kron(sp.eye(ny), kx) \
            + sp.kron(_laplacian_1d(ny, grid.hy, True), sp.eye(nx))
        kv = sp.kron(sp.eye(ny-1), kx) \
            + sp.kron(_laplacian_1d(ny-1, grid.hy, False), sp.eye(nx))
        self.K = sp.block_diag([ku, kv], format='csr')
        dy = sp.diags([np.ones(ny-1), -np.ones(ny-1)], [0, -1],
                      shape=(ny, ny-1), format='csr')/grid.hy
        self.D = sp.hstack([sp.kron(sp.eye(ny), dx), sp.kron(dy, sp.eye(nx))], format='csr')
        self.G = -self.D.T.tocsr()

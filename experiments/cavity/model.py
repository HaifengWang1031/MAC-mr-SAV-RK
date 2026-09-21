"""Stationary side/bottom walls and constant tangential velocity on the top lid."""
import numpy as np
from solver.mac.grid import MACGrid, Array

def lid_viscous_load(grid: MACGrid, nu: float, speed: float) -> Array:
    """Odd extension becomes u_ghost=2*speed-u_top: add 2*nu*speed/hy²."""
    if not np.isfinite([nu,speed]).all() or nu<=0:
        raise ValueError('Finite speed and positive viscosity required')
    u=np.zeros((grid.ny,grid.nx+1))
    v=np.zeros((grid.ny+1,grid.nx))
    u[-1,1:-1]=2*nu*speed/grid.hy**2
    return grid.pack(u,v)

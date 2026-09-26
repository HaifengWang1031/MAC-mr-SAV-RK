"""No-slip analogue of periodic Kolmogorov forcing on (0, 2*pi)^2."""
import numpy as np
from solver.mac.grid import MACGrid, Array


def initial_velocity(grid: MACGrid, epsilon: float = 4., modes: int = 10) -> Array:
    """Discrete curl of a wall-clamped, weighted trigonometric streamfunction."""
    x, y = grid.coordinates('vertex')
    shape = np.zeros_like(x)
    for k1 in range(-modes, modes+1):
        for k2 in range(-modes, modes+1):
            squared = k1*k1+k2*k2
            if 0 < squared <= modes*modes:
                shape += ((np.cos(k1*x)+np.sin(k1*x)) *
                          (np.cos(k2*y)+np.sin(k2*y))) / squared**1.5
    weight = np.sin(np.pi*x/grid.lx)**2 * np.sin(np.pi*y/grid.ly)**2
    psi = epsilon*weight*shape
    psi[[0,-1],:] = 0.
    psi[:,[0,-1]] = 0.
    u = np.diff(psi, axis=0)/grid.hy
    v = -np.diff(psi, axis=1)/grid.hx
    return grid.pack(u,v)


def force(grid: MACGrid, m: int = 4) -> Array:
    """f=(-sin(m*y),0), so curl(f)=m*cos(m*y)."""
    _, y = grid.coordinates('u')
    u = -np.sin(m*y)
    v = np.zeros((grid.ny+1,grid.nx))
    u[:,[0,-1]] = 0.
    return grid.pack(u,v)


def vorticity(grid: MACGrid, velocity: Array) -> Array:
    """Cell-area weighted curl on interior MAC vertices; wall values are excluded."""
    u,v = grid.unpack(velocity)
    return ((v[1:-1,1:]-v[1:-1,:-1])/grid.hx -
            (u[1:,1:-1]-u[:-1,1:-1])/grid.hy)


def vorticity_norm(grid: MACGrid, velocity: Array) -> float:
    curl = vorticity(grid,velocity)
    return float(np.sqrt(grid.area*np.sum(curl*curl)))

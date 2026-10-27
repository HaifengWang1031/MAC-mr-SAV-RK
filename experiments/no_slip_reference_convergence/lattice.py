"""Deterministic no-slip lattice streamfunction on the square (0, 2*pi)^2."""
import numpy as np

from solver.mac.grid import Array, MACGrid


def streamfunction(x: Array, y: Array, *, cutoff: int = 100,
                   amplitude: float = 4.) -> Array:
    """Evaluate the full signed integer-lattice sum on a tensor grid [y, x]."""
    if type(cutoff) is not int or cutoff < 1:
        raise ValueError('Positive integer lattice cutoff required')
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('Positive finite streamfunction amplitude required')
    radius = int(np.floor(np.sqrt(cutoff)))
    modes = np.arange(-radius, radius+1)
    squared = modes[:, None]**2 + modes[None, :]**2
    coefficients = np.zeros(squared.shape, dtype=float)
    active = (squared > 0) & (squared <= cutoff)
    coefficients[active] = squared[active].astype(float)**(-1.5)
    phase_x = modes[:, None]*x[None, :]
    phase_y = modes[:, None]*y[None, :]
    basis_x = np.cos(phase_x)+np.sin(phase_x)
    basis_y = np.cos(phase_y)+np.sin(phase_y)
    total = basis_y.T@coefficients@basis_x
    return amplitude*np.sin(y[:, None]/2.)**2*np.sin(x[None, :]/2.)**2*total


def initial_velocity(grid: MACGrid, *, cutoff: int = 100,
                     amplitude: float = 4.) -> Array:
    """Discrete curl of vertex psi; preserve its amplitude and exact wall zeros."""
    if type(cutoff) is not int or cutoff < 1:
        raise ValueError('Positive integer lattice cutoff required')
    if not np.isclose([grid.lx, grid.ly], 2.*np.pi, rtol=0., atol=1e-12).all():
        raise ValueError('Lattice streamfunction requires the (0, 2*pi)^2 domain')
    if min(grid.nx, grid.ny) <= 2*(np.floor(np.sqrt(cutoff))+1):
        raise ValueError('Grid does not resolve the lattice and wall-weight frequencies')
    x: Array = np.arange(grid.nx+1, dtype=float)*grid.hx
    y: Array = np.arange(grid.ny+1, dtype=float)*grid.hy
    psi = streamfunction(x, y, cutoff=cutoff, amplitude=amplitude)
    psi[:, [0, -1]] = 0.
    psi[[0, -1], :] = 0.
    u = np.diff(psi, axis=0)/grid.hy
    v = -np.diff(psi, axis=1)/grid.hx
    return grid.pack(u, v)


def force_vector(grid: MACGrid, amplitude: float = 1.) -> Array:
    """Sample the steady vertical body force (0, amplitude*sin(x))."""
    x, _ = grid.coordinates('v')
    u = np.zeros((grid.ny, grid.nx+1))
    v = amplitude*np.sin(x)
    v[[0, -1], :] = 0.
    return grid.pack(u, v)


def clamped_force_vector(grid: MACGrid, amplitude: float = 1.) -> Array:
    """Sample C*((1-cos(x))*sin(y), -sin(x)*(1-cos(y))) on MAC faces.

    The continuous force is the curl of C*(1-cos(x))*(1-cos(y)); it is
    divergence-free and has full zero trace on (0, 2*pi)^2. These are analytic
    face samples, without normalization or an additional discrete projection.
    """
    if not np.isclose([grid.lx, grid.ly], 2.*np.pi, rtol=0., atol=1e-12).all():
        raise ValueError('Clamped force requires the (0, 2*pi)^2 domain')
    if not np.isfinite(amplitude):
        raise ValueError('Finite force amplitude required')
    xu, yu = grid.coordinates('u')
    xv, yv = grid.coordinates('v')
    u = amplitude*(1.-np.cos(xu))*np.sin(yu)
    v = -amplitude*np.sin(xv)*(1.-np.cos(yv))
    u[:, [0, -1]] = 0.
    v[[0, -1], :] = 0.
    return grid.pack(u, v)

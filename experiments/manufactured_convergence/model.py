"""Discrete curl of the requested no-slip streamfunction and its exact MAC forcing."""
import numpy as np
from solver.mac.grid import MACGrid, Array
from solver.mac_ns import MACNavierStokes


def spatial_velocity(grid: MACGrid) -> Array:
    """W_h = curl_h[sin²(pi x) sin²(2 pi y)], exactly discrete solenoidal."""
    if grid.lx != 1. or grid.ly != 1.:
        raise ValueError('The manufactured streamfunction is defined on the unit square')
    x, y = grid.coordinates('vertex')
    psi = np.sin(np.pi*x)**2 * np.sin(2*np.pi*y)**2
    u = np.diff(psi,axis=0)/grid.hy
    v = -np.diff(psi,axis=1)/grid.hx
    u[:,[0,-1]] = 0.
    v[[0,-1],:] = 0.
    result=grid.pack(u,v)
    if np.max(np.abs(grid.unpack(result)[0])) == 0:
        raise ValueError('Grid does not resolve the manufactured streamfunction')
    return result


def amplitude(t: float, scale: float) -> float:
    return float(scale*np.exp(-t)*(1+.25*np.sin(np.pi*t)))


def amplitude_derivative(t: float, scale: float) -> float:
    return float(scale*np.exp(-t)*(.25*np.pi*np.cos(np.pi*t)-1-.25*np.sin(np.pi*t)))


def configure(model: MACNavierStokes, scale: float) -> Array:
    """Set f_h so a(t) W_h solves the semidiscrete MAC momentum equation with p=0."""
    shape=spatial_velocity(model.grid)
    stiffness=model.apply_K(shape)
    convection=model.nonlinear(shape)
    model.force=lambda t: (amplitude_derivative(t,scale)*shape
                           +model.nu*amplitude(t,scale)*stiffness
                           +amplitude(t,scale)**2*convection)
    return amplitude(0.,scale)*shape


def exact(grid: MACGrid, t: float, scale: float) -> Array:
    return amplitude(t,scale)*spatial_velocity(grid)

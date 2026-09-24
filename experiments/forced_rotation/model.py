"""The two steady forcings of the bounded-domain torque study, on Omega=(0,2*pi)^2.

`f_A = amplitude*(sin(k*y), -sin(k*x))` carries a net torque about the domain centre, while
`f_B = amplitude*(cos(k*y), -cos(k*x))` carries none. On the `2*pi` square with integer `k`
the torque of A is exactly `2*amplitude/k` and B's is zero, which is the closed form the
tests check against.

`k` counts the wavelengths across the domain, so `k=5` reproduces the published forcing;
amplitude, viscosity, grid and step size are the exploration parameters. Both forces are
steady, are sampled only at the free faces, and drive both velocity components -- that is
what makes the net torque, and the large-scale rotation it produces, observable at all.
"""
import numpy as np
from solver.mac.grid import MACGrid, Array

KINDS=('A','B')

def force_ab(grid: MACGrid, kind: str, amplitude: float = 0.1, k: float = 5.) -> Array:
    """Packed steady body force of kind `A` (net torque) or `B` (zero torque)."""
    if kind not in KINDS: raise ValueError(f'Unknown forcing kind {kind}')
    if not np.isfinite([amplitude,k]).all() or k<=0: raise ValueError('Invalid forcing amplitude or wavenumber')
    x_u,y_u=grid.coordinates('u'); x_v,y_v=grid.coordinates('v')
    force_u=np.zeros((grid.ny,grid.nx+1)); force_v=np.zeros((grid.ny+1,grid.nx))
    if kind=='A':
        force_u[:,1:-1]=amplitude*np.sin(k*y_u[:,1:-1])
        force_v[1:-1,:]=-amplitude*np.sin(k*x_v[1:-1,:])
    else:
        force_u[:,1:-1]=amplitude*np.cos(k*y_u[:,1:-1])
        force_v[1:-1,:]=-amplitude*np.cos(k*x_v[1:-1,:])
    return grid.pack(force_u,force_v)

def torque(grid: MACGrid, force: Array) -> float:
    """`(1/|Omega|) int (r-r_c) x f dA` of a packed face force, the quantity that separates A from B."""
    f_u,f_v=grid.unpack(force)
    centre_u,centre_v=0.5*(f_u[:,:-1]+f_u[:,1:]),0.5*(f_v[:-1,:]+f_v[1:,:])
    x,y=grid.coordinates('p')
    return float(grid.area*np.sum((x-0.5*grid.lx)*centre_v-(y-0.5*grid.ly)*centre_u)/(grid.lx*grid.ly))

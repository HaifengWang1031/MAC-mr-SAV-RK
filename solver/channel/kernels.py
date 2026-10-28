"""Skew MAC dual-cell fluxes: periodic x, odd tangential y extension."""
import numpy as np
from numba import njit

from ..mac.grid import Array
from ..mac.kernels import divergence


@njit(cache=True)
def convection(u: Array, v: Array, hx: float, hy: float) -> tuple[Array, Array]:
    ny, nxp = u.shape
    nx = nxp-1
    cu, cv = np.zeros_like(u), np.zeros_like(v)
    for j in range(ny):
        for i in range(nx):
            im = (i-1) % nx
            east = .5*(u[j, i]+u[j, i+1])
            west = .5*(u[j, i]+u[j, im])
            north = .5*(v[j+1, im]+v[j+1, i])
            south = .5*(v[j, im]+v[j, i])
            un = u[j+1, i] if j < ny-1 else -u[j, i]
            us = u[j-1, i] if j > 0 else -u[j, i]
            cu[j, i] = (east*.5*(u[j, i]+u[j, i+1])
                        -west*.5*(u[j, i]+u[j, im]))/hx
            cu[j, i] += (north*.5*(u[j, i]+un)-south*.5*(u[j, i]+us))/hy
            cu[j, i] -= .5*u[j, i]*((east-west)/hx+(north-south)/hy)
    cu[:, -1] = cu[:, 0]
    for j in range(1, ny):
        for i in range(nx):
            ip, im = (i+1) % nx, (i-1) % nx
            east = .5*(u[j-1, i+1]+u[j, i+1])
            west = .5*(u[j-1, i]+u[j, i])
            north = .5*(v[j, i]+v[j+1, i])
            south = .5*(v[j, i]+v[j-1, i])
            cv[j, i] = (east*.5*(v[j, i]+v[j, ip])
                        -west*.5*(v[j, i]+v[j, im]))/hx
            cv[j, i] += (north*.5*(v[j, i]+v[j+1, i])
                         -south*.5*(v[j, i]+v[j-1, i]))/hy
            cv[j, i] -= .5*v[j, i]*((east-west)/hx+(north-south)/hy)
    return cu, cv


def warmup() -> float:
    """Measure compilation/cache loading separately from integration."""
    from time import perf_counter
    start = perf_counter()
    u, v = np.zeros((3, 4)), np.zeros((4, 3))
    convection(u, v, 1., 1.)
    divergence(u, v, 1., 1.)
    return perf_counter()-start

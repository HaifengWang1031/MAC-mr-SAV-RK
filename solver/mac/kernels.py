"""Centered dual-cell MAC fluxes. No fastmath; odd tangential wall extension."""
import numpy as np
from numba import njit
from .grid import Array

@njit(cache=True)
def convection(u: Array, v: Array, hx: float, hy: float) -> tuple[Array, Array]:
    ny,nxp=u.shape
    nx=nxp-1
    cu=np.zeros_like(u)
    cv=np.zeros_like(v)
    for j in range(ny):
        for i in range(1,nx):
            east=0.5*(u[j,i]+u[j,i+1])
            west=0.5*(u[j,i]+u[j,i-1])
            north=0.5*(v[j+1,i-1]+v[j+1,i])
            south=0.5*(v[j,i-1]+v[j,i])
            un=u[j+1,i] if j<ny-1 else -u[j,i]
            us=u[j-1,i] if j>0 else -u[j,i]
            cu[j,i]=(east*0.5*(u[j,i]+u[j,i+1])-west*0.5*(u[j,i]+u[j,i-1]))/hx
            cu[j,i]+=(north*0.5*(u[j,i]+un)-south*0.5*(u[j,i]+us))/hy
            # Conservative minus half dual divergence is skew even off ker(D).
            cu[j,i]-=0.5*u[j,i]*((east-west)/hx+(north-south)/hy)
    for j in range(1,ny):
        for i in range(nx):
            east=0.5*(u[j-1,i+1]+u[j,i+1])
            west=0.5*(u[j-1,i]+u[j,i])
            north=0.5*(v[j,i]+v[j+1,i])
            south=0.5*(v[j,i]+v[j-1,i])
            ve=v[j,i+1] if i<nx-1 else -v[j,i]
            vw=v[j,i-1] if i>0 else -v[j,i]
            cv[j,i]=(east*0.5*(v[j,i]+ve)-west*0.5*(v[j,i]+vw))/hx
            cv[j,i]+=(north*0.5*(v[j,i]+v[j+1,i])-south*0.5*(v[j,i]+v[j-1,i]))/hy
            cv[j,i]-=0.5*v[j,i]*((east-west)/hx+(north-south)/hy)
    return cu,cv

@njit(cache=True)
def divergence(u: Array,v: Array,hx: float,hy: float) -> Array:
    ny,nxp=u.shape
    result=np.empty((ny,nxp-1))
    for j in range(ny):
        for i in range(nxp-1):
            result[j,i]=(u[j,i+1]-u[j,i])/hx+(v[j+1,i]-v[j,i])/hy
    return result

@njit(cache=True)
def inner_faces(u: Array,v: Array,a: Array,b: Array,cell_area: float) -> float:
    total=0.0
    for j in range(u.shape[0]):
        for i in range(1,u.shape[1]-1): total+=u[j,i]*a[j,i]
    for j in range(1,v.shape[0]-1):
        for i in range(v.shape[1]): total+=v[j,i]*b[j,i]
    return cell_area*total

def warmup() -> float:
    """Measure first dispatch/compilation separately from production stepping."""
    from time import perf_counter
    start=perf_counter()
    u=np.zeros((3,4));v=np.zeros((4,3))
    convection(u,v,1.0,1.0)
    divergence(u,v,1.0,1.0)
    inner_faces(u,v,u,v,1.0)
    return perf_counter()-start

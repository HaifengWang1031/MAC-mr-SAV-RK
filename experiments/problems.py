"""Smooth no-slip manufactured fields, used as an independent reference by the tests.

No experiment kind drives these fields any more (the `stokes_mms`/`ns_mms`/`decay` entry
points were removed 2026-09-23); `tests/test_numerics.py` and `tools/benchmark_steps.py`
are the remaining callers, which is why they stay outside the generic solver.
"""
from functools import lru_cache
import numpy as np
import sympy as sy
from solver.mac.grid import MACGrid, Array


def initial_velocity(grid: MACGrid, amplitude: float = .1, t: float = 0.) -> Array:
    x,y=grid.coordinates('vertex')
    psi=amplitude*np.sin(np.pi*x/grid.lx)**2*np.sin(np.pi*y/grid.ly)**2*np.exp(-t)
    psi[[0,-1],:]=0
    psi[:,[0,-1]]=0
    u=np.diff(psi,axis=0)/grid.hy
    v=-np.diff(psi,axis=1)/grid.hx
    return grid.pack(u,v)

@lru_cache(maxsize=16)
def _expressions(lx: float, ly: float, nu: float):
    x,y,t=sy.symbols('x y t')
    psi=sy.sin(sy.pi*x/lx)**2*sy.sin(sy.pi*y/ly)**2*sy.exp(-t)
    u,v=sy.diff(psi,y),-sy.diff(psi,x)
    p=sy.cos(sy.pi*x/lx)*sy.cos(sy.pi*y/ly)*sy.exp(-t)
    values=[u,v,p]
    for transient in (False,True):
        for q,axis in ((u,x),(v,y)):
            visc=-nu*(sy.diff(q,x,2)+sy.diff(q,y,2))
            # Velocity scales with amplitude; convection scales quadratically.
            values.extend([visc+sy.diff(p,axis)+(sy.diff(q,t) if transient else 0),
                           u*sy.diff(q,x)+v*sy.diff(q,y) if transient else sy.Integer(0)])
    return [sy.lambdify((x,y,t),expr,'numpy') for expr in values]

def exact_fields(grid: MACGrid, nu: float, amplitude: float, t: float) -> tuple[Array,Array]:
    functions=_expressions(grid.lx,grid.ly,nu)
    xu,yu=grid.coordinates('u'); xv,yv=grid.coordinates('v'); xp,yp=grid.coordinates('p')
    u=amplitude*functions[0](xu,yu,t); v=amplitude*functions[1](xv,yv,t)
    u[:,[0,-1]]=0; v[[0,-1],:]=0
    pressure=amplitude*functions[2](xp,yp,t); pressure-=pressure.mean()
    return grid.pack(u,v),pressure

def forcing(grid: MACGrid, nu: float, amplitude: float, t: float, transient: bool = True) -> Array:
    functions=_expressions(grid.lx,grid.ly,nu)
    offset=7 if transient else 3
    xu,yu=grid.coordinates('u'); xv,yv=grid.coordinates('v')
    fu=amplitude*functions[offset](xu,yu,t)+amplitude**2*functions[offset+1](xu,yu,t)
    fv=amplitude*functions[offset+2](xv,yv,t)+amplitude**2*functions[offset+3](xv,yv,t)
    fu[:,[0,-1]]=0; fv[[0,-1],:]=0
    return grid.pack(fu,fv)

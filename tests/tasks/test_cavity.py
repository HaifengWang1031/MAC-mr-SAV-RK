import numpy as np
from solver.mac.grid import MACGrid
from solver.mac.operators import MACOperators
from experiments.cavity.model import lid_viscous_load

def test_lid_load_matches_independent_ghost_stencil():
    g=MACGrid(7,5,1.3,.8); nu=.03; speed=1.2
    z=np.random.default_rng(7).normal(size=g.size)
    u,v=g.unpack(z)
    lap_u=np.zeros_like(u); lap_v=np.zeros_like(v)
    for j in range(g.ny):
        for i in range(1,g.nx):
            north=u[j+1,i] if j<g.ny-1 else 2*speed-u[j,i]
            south=u[j-1,i] if j>0 else -u[j,i]
            lap_u[j,i]=(u[j,i+1]-2*u[j,i]+u[j,i-1])/g.hx**2+(north-2*u[j,i]+south)/g.hy**2
    for j in range(1,g.ny):
        for i in range(g.nx):
            east=v[j,i+1] if i<g.nx-1 else -v[j,i]
            west=v[j,i-1] if i>0 else -v[j,i]
            lap_v[j,i]=(east-2*v[j,i]+west)/g.hx**2+(v[j+1,i]-2*v[j,i]+v[j-1,i])/g.hy**2
    actual=-nu*(MACOperators(g).K@z)+lid_viscous_load(g,nu,speed)
    np.testing.assert_allclose(actual,nu*g.pack(lap_u,lap_v),rtol=1e-13,atol=1e-12)


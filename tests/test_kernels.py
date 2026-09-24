import numpy as np
from solver.mac.grid import MACGrid
from solver.mac.kernels import convection

def numpy_flux_reference(u,v,hx,hy):
    # Vectorized dual-cell fluxes, independent of the loop implementation.
    up=np.pad(u,((1,1),(0,0)))
    up[0]=-u[0]; up[-1]=-u[-1]
    vp=np.pad(v,((0,0),(1,1)))
    vp[:,0]=-v[:,0]; vp[:,-1]=-v[:,-1]
    center=u[:,1:-1]
    e=(center+u[:,2:])/2; w=(center+u[:,:-2])/2
    n=(v[1:,:-1]+v[1:,1:])/2; s=(v[:-1,:-1]+v[:-1,1:])/2
    cu=(e*(center+u[:,2:])/2-w*(center+u[:,:-2])/2)/hx
    cu+=(n*(center+up[2:,1:-1])/2-s*(center+up[:-2,1:-1])/2)/hy
    cu-=center*((e-w)/hx+(n-s)/hy)/2
    center=v[1:-1,:]
    e=(u[:-1,1:]+u[1:,1:])/2; w=(u[:-1,:-1]+u[1:,:-1])/2
    n=(center+v[2:,:])/2; s=(center+v[:-2,:])/2
    cv=(e*(center+vp[1:-1,2:])/2-w*(center+vp[1:-1,:-2])/2)/hx
    cv+=(n*(center+v[2:,:])/2-s*(center+v[:-2,:])/2)/hy
    cv-=center*((e-w)/hx+(n-s)/hy)/2
    outu=np.zeros_like(u); outv=np.zeros_like(v)
    outu[:,1:-1]=cu; outv[1:-1,:]=cv
    return outu,outv

def test_numba_matches_independent_numpy_reference():
    from solver.mac.kernels import divergence,inner_faces
    from solver.mac.operators import MACOperators
    grid=MACGrid(11,7,2.,.7)
    x=np.random.default_rng(73).normal(size=grid.size)
    u,v=grid.unpack(x)
    expected=numpy_flux_reference(u,v,grid.hx,grid.hy)
    actual=convection(u,v,grid.hx,grid.hy)
    for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(divergence(u,v,grid.hx,grid.hy).ravel(),MACOperators(grid).D@x,atol=1e-13)
    assert abs(inner_faces(u,v,u,v,grid.area)-grid.inner(x,x))<1e-12

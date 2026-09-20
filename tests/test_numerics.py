import numpy as np
import pytest
from solver.mac.grid import MACGrid
from solver.mac.operators import MACOperators
from solver.mac.stokes import DirectStokes
from solver.mac_ns import MACNavierStokes
from solver.schemes.sdirk2 import SDIRK2, ETA
from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV
from solver.integrate import integrate
from experiments.problems import initial_velocity,forcing,exact_fields

def test_stokes_spatial_convergence_and_rectangular_structure():
    errors=[]
    for nx,ny in ((8,6),(16,12),(32,24)):
        g=MACGrid(nx,ny,1.3,.8); ops=MACOperators(g)
        assert (ops.G+ops.D.T).nnz==0
        assert (ops.K-ops.K.T).nnz==0
        x=np.random.default_rng(3).normal(size=g.size)
        assert g.inner(x,ops.K@x)>0
        solved=DirectStokes(ops).solve(forcing(g,.1,.1,0,False),mass=0.,viscosity=.1)
        exact,p=exact_fields(g,.1,.1,0)
        errors.append([g.norm(solved.velocity-exact),np.sqrt(g.area*np.sum((solved.pressure-p)**2))])
    rates=np.log2(np.array(errors[:-1])/np.array(errors[1:]))
    assert np.min(rates)>1.7

@pytest.mark.parametrize('scheme',[SDIRK2(),SDIRK2MRSAV(gamma=1.)])
def test_second_order_against_exact_semidiscrete_solution(scheme):
    g=MACGrid(10,8); m=MACNavierStokes(g,.1)
    base=initial_velocity(g,.05)
    def f(t):
        z=np.exp(-t)*base
        return -z+m.nu*(m.ops.K@z)+m.nonlinear(z)
    m.force=f
    errors=[]
    for count in (10,20,40):
        result=integrate(m,scheme,m.state(0,base),[.1/count]*count)
        assert result.status=='complete'
        errors.append(g.norm(m.vector(result.final)-np.exp(-.1)*base))
    assert min(np.log2(np.array(errors[:-1])/errors[1:]))>1.85

def test_sav_discrete_energy_balance():
    g=MACGrid(10,8); m=MACNavierStokes(g,.03); scheme=SDIRK2MRSAV(2.)
    old=initial_velocity(g,.8); dt=.02; r0=.05
    n0=m.nonlinear(old)
    trial=scheme.step(m,m.state(0.,old,r0),dt)
    s1,s2=trial.stages
    first=m.backend.solve(old-dt*ETA*(1-s1.r*s1.r)*n0,mass=1.,viscosity=m.nu*ETA*dt).velocity
    second=m.vector(trial.state)
    balance=0.
    velocities=[old,first,second]; scalars=[r0,s1.r,s2.r]
    for i in (1,2):
        balance+=g.inner(velocities[i]-velocities[i-1],velocities[i])+(scalars[i]-scalars[i-1])*(scalars[i]-1)
        for j in range(1,i+1):
            a=ETA if i==j else 1-2*ETA
            balance+=dt*a*(m.nu*g.inner(m.ops.K@velocities[j],velocities[i])+2.*scalars[j]*(scalars[i]-1))
    assert abs(balance)<1e-11

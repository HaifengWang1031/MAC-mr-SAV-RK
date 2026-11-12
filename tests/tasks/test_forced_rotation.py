"""A/B torque forcing, rotation observables and reversal detection."""
import numpy as np
import pytest
from experiments.forced_rotation.analyze import detect_reversals, energy_balance_residual, residence_times
from experiments.forced_rotation.model import force_ab, torque
from experiments.workflow import effective_config
from solver.mac.grid import MACGrid
from solver.mac_ns import MACNavierStokes

def test_torque_matches_the_closed_form_for_the_two_forcings():
    # On the 2*pi square with integer k, f_A carries exactly 2*amplitude/k and f_B carries none.
    for kind,expected in (('A',2*0.1/5.),('B',0.)):
        for n in (64,128):
            grid=MACGrid(n,n,2*np.pi,2*np.pi)
            measured=torque(grid,force_ab(grid,kind,0.1,5.))
            assert measured==pytest.approx(expected,abs=5e-4)

def test_forcing_samples_both_components_on_free_faces():
    grid=MACGrid(9,7,2*np.pi,2*np.pi)
    force=force_ab(grid,'A',.1,5.)
    assert force.shape==(grid.size,)
    # A drives both components: neither block may vanish identically.
    assert np.abs(force[:grid.nu]).max()>0 and np.abs(force[grid.nu:]).max()>0
    with pytest.raises(ValueError):force_ab(grid,'C')

def rigid_rotation(grid, omega):
    """No-slip-compatible rotation: the profile is tapered to zero on all four walls."""
    x_u,y_u=grid.coordinates('u'); x_v,y_v=grid.coordinates('v')
    taper_u=np.sin(np.pi*x_u/grid.lx)**2*np.sin(np.pi*y_u/grid.ly)**2
    taper_v=np.sin(np.pi*x_v/grid.lx)**2*np.sin(np.pi*y_v/grid.ly)**2
    u=np.zeros((grid.ny,grid.nx+1)); v=np.zeros((grid.ny+1,grid.nx))
    u[:,1:-1]=-omega*(y_u[:,1:-1]-np.pi)*taper_u[:,1:-1]
    v[1:-1,:]=omega*(x_v[1:-1,:]-np.pi)*taper_v[1:-1,:]
    return grid.pack(u,v)

def test_angular_momentum_converges_to_its_quadrature_reference():
    # The cell-centred quadrature in `angular_momentum` is second order, so the discrete L of
    # one fixed smooth field must approach an independent analytic integral of the same
    # integral; the taper keeps the field no-slip compatible, which is what `pack` assumes.
    omega=.7
    # Exact integral of omega*((x-pi)^2+(y-pi)^2)*sin(x/2)^2*sin(y/2)^2.
    # Integral sin(x/2)^2 dx = pi, and its second central moment is pi^3/3-2*pi.
    reference=omega*(2*np.pi**4/3-4*np.pi**2)
    errors=[]
    for n in (48,96):
        grid=MACGrid(n,n,2*np.pi,2*np.pi)
        errors.append(abs(MACNavierStokes(grid,1e-3).angular_momentum(rigid_rotation(grid,omega))-reference)/abs(reference))
    assert errors[1]<errors[0] and errors[1]<2e-3
    assert np.log2(errors[0]/errors[1])>1.5

def test_angular_momentum_power_and_dissipation_of_a_known_field():
    grid=MACGrid(48,48,2*np.pi,2*np.pi); model=MACNavierStokes(grid,1e-3)
    packed=rigid_rotation(grid,.7)
    diagnostics=model.diagnostics(model.state(0.,packed))
    assert diagnostics['dissipation']==pytest.approx(model.nu*diagnostics['h1_seminorm_squared'],rel=1e-12)
    constant=np.ones(grid.size)
    model.force=lambda t: constant
    assert model.diagnostics(model.state(0.,constant))['power']==pytest.approx(grid.inner(constant,constant),rel=1e-12)

def test_reversal_detection_needs_threshold_and_duration():
    times=np.arange(0.,20.,.1)
    # A clean sign change that holds: one reversal, timed when the new sign leaves the band.
    slow=np.sin(2*np.pi*times/20.)
    events=detect_reversals(times,slow,threshold=.2,min_duration=1.)
    # The reported time is the band exit, not the zero crossing: it lags it by
    # arcsin(threshold/amplitude)*period/(2*pi) = 0.64 here, plus up to one sample.
    assert len(events)==1 and events[0]['from']==1. and events[0]['to']==-1.
    assert events[0]['time']==pytest.approx(10.64,abs=.15)
    # Growing from zero establishes the sign without being a reversal.
    growing=times/10.
    assert detect_reversals(times,growing,threshold=.2,min_duration=1.)==[]
    # Fast jitter about zero never leaves the band, so it is not a reversal.
    jitter=.05*np.sin(2*np.pi*times/1.)
    assert detect_reversals(times,jitter,threshold=.2,min_duration=1.)==[]
    # A spike with the other sign that decays back inside the band is rejected by the duration.
    spike=slow.copy(); spike[100:103]=-3.
    assert len(detect_reversals(times,spike,threshold=.2,min_duration=1.))==1
    with pytest.raises(ValueError):detect_reversals(times,slow,threshold=-1.,min_duration=1.)

def test_residence_times_close_the_record():
    events=[{'time':1.},{'time':4.},{'time':6.}]
    assert residence_times(events,np.array([0.,10.]))==pytest.approx([3.,2.,4.])
    assert residence_times([],np.array([0.,10.]))==[]

def test_energy_balance_residual_is_zero_for_a_consistent_series():
    times=np.arange(0.,2.,.1)
    history={'time':times,'kinetic':np.sin(times),'power':np.cos(times),'dissipation':np.zeros_like(times),
             'angular_momentum':np.zeros_like(times),'h1_seminorm_squared':np.zeros_like(times),'r':np.zeros_like(times)}
    record={'history':history,'config':{'dt':.1}}
    index=np.arange(times.size)
    # dE/dt = P exactly here, so the reported residual must be at quadrature level.
    residual=energy_balance_residual(record,index)
    assert abs(residual['relative_residual'])<2e-2
    assert residual['samples']==times.size

def test_rotation_experiment_config_is_validated():
    base=dict(experiment='rotation_ns',nx=64,ny=64,lx=2*np.pi,ly=2*np.pi,nu=1e-3,amplitude=.1,k_f=5.,
              forcing_kind='A',T=1.,dt=.01)
    cfg=effective_config(base)
    assert cfg['force']=='amplitude*(sin(5.0*y), -sin(5.0*x))'
    assert cfg['initial_condition']=='zero' and cfg['boundary']=='homogeneous_no_slip'
    assert effective_config({**base,'forcing_kind':'B'})['force']=='amplitude*(cos(5.0*y), -cos(5.0*x))'
    with pytest.raises(ValueError,match='Unknown forcing_kind'):effective_config({**base,'forcing_kind':'C'})
    with pytest.raises(ValueError,match='Invalid forcing wavenumber'):effective_config({**base,'k_f':0.})
    with pytest.raises(ValueError,match='only supported for rotation_ns'):
        effective_config({**base,'experiment':'cavity','forcing_kind':'A'})

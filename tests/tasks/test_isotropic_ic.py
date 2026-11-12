"""Clamped-beam basis and the isotropic no-slip initial condition."""
import h5py
import numpy as np
import pytest
from experiments.forced_ns_convergence.isotropic import (beam,build_isotropic,coefficient_table,velocity_at)
from experiments.workflow import load_record,run_experiment
from solver.mac.grid import MACGrid
from solver.mac.operators import MACOperators
from solver.mac.stokes import DirectStokes

# 60-digit references from mpmath (mp.dps=60), pasted so the test needs no arbitrary precision.
EIGENVALUES={1:4.73004074486270403,2:7.85320462409583756,3:10.9956078380016709,
             4:14.1371654912574642,5:17.2787596573994814}
BEAM_REFERENCE={
    5:(1.502383066135728,-0.527889700093016561,-0.784140003443301687,1.41456751133068692),
    12:(-1.16849380798201624,0.541250591035150384,1.38704024754077119,0.0),
    24:(0.785761271194790994,-0.541196095748808521,0.275899379283234616,0.0)}

def test_beam_matches_high_precision_reference():
    # The naive cosh/sinh form is off by 2e-2 at p=12 and 8e-1 at p=24; the delta form must
    # reproduce the reference to near machine precision.
    for p,values in BEAM_REFERENCE.items():
        for index,expected in enumerate(values):
            x=np.array([(index+1)/8.])
            assert beam(p,x)[0]==pytest.approx(expected,abs=1e-11)

def test_isotropic_field_clears_all_four_walls():
    grid=MACGrid(48,32,1.3,.8)
    modes,coefficients=coefficient_table(grid,k_lo=3.,k_hi=4.,alpha=5./3.,seed=1)
    x,y=grid.coordinates('vertex')
    u,v=velocity_at(grid,modes,coefficients,x,y)
    boundary=(x<=0)|(x>=grid.lx)|(y<=0)|(y>=grid.ly)
    assert np.abs(u[boundary]).max()<1e-11
    assert np.abs(v[boundary]).max()<1e-11

def test_isotropic_divergence_is_second_order_and_projection_removes_it():
    # One fixed continuum field sampled on three grids: only the band matters, not the grid.
    errors=[]
    for nx,ny in ((16,16),(32,32),(64,64)):
        grid=MACGrid(nx,ny)
        ops=MACOperators(grid)
        packed=build_isotropic(grid,k_lo=1.5,k_hi=2.,seed=2)
        scale=float(np.max(np.abs(packed)))
        errors.append(float(np.max(np.abs(ops.D@packed)))/scale)
        projected=DirectStokes(ops).solve(packed,mass=1.,viscosity=0.).velocity
        assert float(np.max(np.abs(ops.D@projected)))<1e-10
    rates=np.log2(np.array(errors[:-1])/np.array(errors[1:]))
    assert rates.min()>1.5
    assert errors[-1]<.3

def test_isotropic_field_is_statistically_isotropic():
    grid=MACGrid(64,64)
    packed=build_isotropic(grid,k_lo=4.,k_hi=8.,seed=0)
    u,v=grid.unpack(packed)
    uu=grid.area*float(np.dot(packed[:grid.nu],packed[:grid.nu]))
    vv=grid.area*float(np.dot(packed[grid.nu:],packed[grid.nu:]))
    uv=grid.area*float(np.mean(0.5*(u[:,:-1]+u[:,1:])*0.5*(v[:-1,:]+v[1:,:])))
    assert abs(uv)/(uu+vv)<1e-4
    assert abs(uu-vv)/(uu+vv)<.05
    symmetric=build_isotropic(grid,k_lo=4.,k_hi=8.,seed=0,symmetric=True)
    us,vs=grid.unpack(symmetric)
    uus=grid.area*float(np.dot(symmetric[:grid.nu],symmetric[:grid.nu]))
    vvs=grid.area*float(np.dot(symmetric[grid.nu:],symmetric[grid.nu:]))
    assert abs(uus-vvs)/(uus+vvs)<1e-12

def test_forced_ns_run_carries_the_isotropic_initial_condition(tmp_path):
    base={'experiment':'forced_ns','nx':16,'ny':16,'nu':.01,'amplitude':1.,'m':1,'T':.02,'dt':.005,
          'snapshots':[.02],'ic_kind':'isotropic_beams','ic_k_lo':1.,'ic_k_hi':2.,'ic_value':1.,'log_every':1}
    first=run_experiment(base,root=tmp_path)
    config,record=load_record(first,require_complete=True)
    assert 'isotropic_beams' in config['initial_condition'] and config['m']==1
    assert record['status']=='complete'
    assert run_experiment(base,root=tmp_path)==first
    # A different seed must give a different flow, so the initial condition really enters.
    other=run_experiment({**base,'ic_seed':1},root=tmp_path)
    assert other!=first
    with h5py.File(other/'results.h5') as changed, h5py.File(first/'results.h5') as reference:
        assert not np.array_equal(changed['final/u'][:],reference['final/u'][:])

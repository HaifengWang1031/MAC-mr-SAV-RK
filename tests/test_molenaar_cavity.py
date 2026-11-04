"""Independent source/curl, backend equivalence and streamed public workflow checks."""
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from experiments.molenaar_cavity.model import force_field, make_model
from experiments.molenaar_cavity.run import run_case
from solver.mac.grid import MACGrid
from solver.schemes.sdirk3 import SDIRK3


def config() -> dict:
    return dict(n=12,A0=.05,kappa=6*np.pi,q_spin=.01,nu_prime=40.,backend='direct',
                dt=.001,T=.004,sample_every=.001,snapshot_every=.002,log_seconds=30.)


def test_source_curl_matches_independent_staggered_formula():
    g=MACGrid(48,36,2.,2.)
    k=6*np.pi
    fu,fv=g.unpack(force_field(g,A0=.05,kappa=k,q_spin=.02))
    curl=np.diff(fv[1:-1,:],axis=1)/g.hx-np.diff(fu[:,1:-1],axis=0)/g.hy
    x,y=g.coordinates('vertex'); x=x[1:-1,1:-1]-1; y=y[1:-1,1:-1]-1
    expected=.1*np.sinc(k*g.hx/(2*np.pi))*np.cos(k*x)+.1*np.sinc(k*g.hy/(2*np.pi))*np.cos(k*y)+.02
    np.testing.assert_allclose(curl,expected,atol=2e-15)


def test_sdirk3_candidate_tensor_matches_direct():
    c=config()
    fields=[]
    for backend in ('direct','tensor'):
        m=make_model({**c,'backend':backend})
        state=m.state(0.,np.zeros(m.grid.size))
        trial=SDIRK3().step(m,state,c['dt'])
        assert max(s.divergence_inf for s in trial.stages)<1e-9
        fields.append(m.vector(trial.state))
    np.testing.assert_allclose(*fields,atol=2e-12,rtol=2e-10)


@pytest.mark.parametrize('scheme,r0',[('sdirk3',0.),('sdirk3_mrsav',.2)])
def test_streamed_run_retains_config_fields_probe_and_status(tmp_path: Path, scheme: str, r0: float):
    output=tmp_path/'run'
    outcome=run_case({**config(),'T':.02,'scheme':scheme,'gamma':1.,'r0':r0},str(output))
    assert outcome['status']=='complete'
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['config']['kappa']==6*np.pi
    assert manifest['config']['q_spin']==.01
    assert manifest['config']['scheme']==scheme
    with h5py.File(output/'results.h5') as h:
        assert h.attrs['status']=='complete'
        np.testing.assert_allclose(h['diagnostics'][:,0],np.arange(21)*.001)
        assert h['diagnostics'].dtype==np.dtype('f8')
        np.testing.assert_allclose(h['snapshots/t'][:],np.arange(11)*.002)
        assert h['final/u'].shape==(12,13)
        assert h['final'].attrs['t']==.02
        assert np.isfinite(h['diagnostics'][:]).all()
        assert h['stage_records'].shape==(20,4,16)
        np.testing.assert_allclose(h['snapshots/r'][-1],h['final'].attrs['r'])
        if scheme=='sdirk3_mrsav':
            assert .19 < h['final'].attrs['r'] < .2
            assert np.all(h['stage_records'][:,:,1] == 0)
            assert np.all(h['scalar_corrections'][:,:,0] == 2)
            np.testing.assert_array_equal(h['scalar_corrections'][:,:,3],h['stage_records'][:,:,0])
            assert np.max(h['stage_records'][:,:,12])<1e-8

    from experiments.molenaar_cavity.analyze import analyze
    report=analyze(output,tmp_path/'report',discard=0.)
    assert report['samples']==21
    assert (tmp_path/'report'/'spectrum.npz').exists()

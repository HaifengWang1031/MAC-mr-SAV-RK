"""Four-stage persistence and experiment campaign/reference labels."""
import h5py
import pytest

from experiments.workflow import run_experiment


@pytest.mark.parametrize('scheme', ['sdirk3','sdirk3_mrsav'])
def test_four_stages_are_saved_and_prescribed_steps_work(tmp_path,scheme):
    path=run_experiment({'experiment':'forced_ns','scheme':scheme,'nx':8,'ny':6,
                         'T':.02,'steps':[.007,.013],'m':1},root=tmp_path)
    with h5py.File(path/'results.h5') as data:
        assert data.attrs['status']=='complete'
        assert data['stages/residual'].shape==(2,4)
        assert data['roots/candidates'].shape==(2,4,5)
        assert data['final/stage_pressure'].shape[0]==4


def test_sdirk3_campaign_uses_sdirk3_reference_and_labels(tmp_path):
    import json
    from experiments.forced_ns_convergence.run import run_campaign
    cfg={'base':{'experiment':'forced_ns','nx':8,'ny':8,'m':1,'T':.04,
                 'snapshots':[.04],'log_every':100},
         'tau_base':.08,'k_levels':[1,2],'reference_k':4,
         'sensitivity_threshold':.5,'max_reference_refinements':0,
         'schemes':['sdirk3','sdirk3_mrsav']}
    batch_path=run_campaign(cfg,root=tmp_path)
    batch=json.loads(batch_path.read_text())
    assert [entry['scheme'] for entry in batch['trials']]==['sdirk3','sdirk3_mrsav']*2
    assert batch['references']['reference_k']==4
    report=__import__('pathlib').Path(batch['reports'][0])
    latex=(report/'tables/L2.tex').read_text()
    assert 'SDIRK3-mrSAV' in latex and 'order 2' not in latex

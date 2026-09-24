from experiments.analysis import analyze_runs
import json
import h5py
from experiments.workflow import run_experiment

def test_save_reuse_rerun_and_analysis(tmp_path):
    config={'experiment':'cavity','nx':8,'ny':6,'T':.02,'dt':.01,'scheme':'sdirk2_mrsav'}
    first=run_experiment(config,root=tmp_path)
    assert run_experiment(config,root=tmp_path)==first
    second=run_experiment(config,root=tmp_path,rerun=True)
    assert second!=first
    manifest=json.loads((first/'manifest.json').read_text())
    assert manifest['status']=='complete'
    with h5py.File(first/'results.h5') as data:
        assert data['times'].shape==(3,)
        assert data['roots/candidates'].shape==(2,2,3)
        assert data['final/u'].shape==(6,9)
    report=analyze_runs([first],root=tmp_path)
    assert (report/'figures/diagnostics.png').exists()
    assert json.loads((report/'analysis.json').read_text())['status']=='complete'
    (first/'results.h5').write_bytes(b'corrupted')
    # Explicit rerun remains a separate attempt; corrupted first result cannot be reused.
    assert run_experiment(config,root=tmp_path)==second

def test_failed_run_is_saved_and_not_reused(tmp_path,monkeypatch):
    from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV
    original=SDIRK2MRSAV.step
    def fail_second(self,model,state,dt):
        if state.t>0: raise RuntimeError('failure recording test')
        return original(self,model,state,dt)
    monkeypatch.setattr(SDIRK2MRSAV,'step',fail_second)
    config={'experiment':'cavity','nx':6,'ny':4,'T':.02,'dt':.01}
    first=run_experiment(config,root=tmp_path)
    assert json.loads((first/'manifest.json').read_text())['status']=='failed'
    with h5py.File(first/'results.h5') as data:
        assert data['final/t'][()]==.01
        assert data['times'].shape==(2,)
    assert run_experiment(config,root=tmp_path)!=first


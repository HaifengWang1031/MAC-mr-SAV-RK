from experiments.analysis import analyze_runs
import json
import h5py
from experiments.workflow import run_experiment

def test_save_reuse_rerun_and_analysis(tmp_path):
    config={'experiment':'decay','nx':8,'ny':6,'T':.02,'dt':.01,'scheme':'sdirk2_mrsav'}
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
    config={'experiment':'decay','nx':6,'ny':4,'T':.02,'dt':.01}
    first=run_experiment(config,root=tmp_path)
    assert json.loads((first/'manifest.json').read_text())['status']=='failed'
    with h5py.File(first/'results.h5') as data:
        assert data['final/t'][()]==.01
        assert data['times'].shape==(2,)
    assert run_experiment(config,root=tmp_path)!=first

def test_prescribed_batch_records_members(tmp_path):
    from experiments.cli import run_batch
    batch=run_batch({'base':{'experiment':'decay','nx':6,'ny':4,'T':.03,'steps':[.01,.02]},
                     'cases':[{'scheme':'sdirk2'},{'scheme':'sdirk2_mrsav'}]},root=tmp_path)
    record=json.loads((batch/'batch.json').read_text())
    assert record['status']=='complete'
    assert len(record['members'])==2
    for member in record['members']:
        assert (tmp_path/member['path']/'results.h5').exists()

def test_analysis_rejects_modified_config(tmp_path):
    import pytest
    path=run_experiment({'experiment':'decay','nx':5,'ny':4,'T':.01,'dt':.01},root=tmp_path)
    cfg=json.loads((path/'config.json').read_text()); cfg['nu']=1.
    (path/'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='checksum'):
        analyze_runs([path],root=tmp_path)

def test_temporal_plot_failure_marks_report_failed(tmp_path,monkeypatch):
    import pytest
    from matplotlib.figure import Figure
    from experiments.convergence import analyze_temporal
    paths=[run_experiment({'experiment':'ns_mms','nx':6,'ny':4,'T':.04,'dt':dt},root=tmp_path)
           for dt in (.02,.01,.005,.0025)]
    save=Figure.savefig
    def broken(self,path,*args,**kwargs):
        if str(path).endswith('time_convergence.png'): raise OSError('intentional plot failure')
        return save(self,path,*args,**kwargs)
    monkeypatch.setattr(Figure,'savefig',broken)
    with pytest.raises(OSError,match='intentional'):
        analyze_temporal(paths[:2],paths[2],paths[3],root=tmp_path)
    records=list((tmp_path/'reports').glob('*/*/analysis.json'))
    assert len(records)==1
    assert json.loads(records[0].read_text())['status']=='failed'

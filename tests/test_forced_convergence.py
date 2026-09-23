import json
from pathlib import Path
import numpy as np
import pytest
from experiments.forced_ns_convergence.model import force_vector
from experiments.forced_ns_convergence.run import run_campaign, validate
from experiments.forced_ns_convergence.analyze import analyze_batch
from solver.mac.grid import MACGrid
from solver.mac_ns import MACNavierStokes
from solver.integrate import integrate


def test_force_samples_only_free_faces():
    grid=MACGrid(7,5,1.3,.8)
    force=force_vector(grid)
    assert np.all(force[:grid.nu]==0.)
    expected=np.tile(np.sin((np.arange(grid.nx)+.5)*grid.hx),grid.ny-1)
    np.testing.assert_allclose(force[grid.nu:],expected)


def test_long_fixed_schedule_does_not_reject_final_snapshot():
    class FailImmediately:
        name='intentional'
        def step(self,*args):raise RuntimeError('deliberate stop')
    model=MACNavierStokes(MACGrid(3,3),.1)
    result=integrate(model,FailImmediately(),model.state(0.,np.zeros(model.grid.size)),
                     np.full(327680,.1*2**-14),snapshots=[2.])
    assert result.status=='failed' and 'deliberate stop' in result.error


def test_campaign_reuse_and_explicit_analysis(tmp_path):
    config=json.loads(Path('experiments/forced_ns_convergence/configs/smoke.json').read_text())
    first=run_campaign(config,root=tmp_path)
    second=run_campaign(config,root=tmp_path)
    a,b=(json.loads(p.read_text()) for p in (first,second))
    assert a['status']==b['status']=='complete'
    assert a['trials']==b['trials'] and a['references']==b['references']
    report=analyze_batch(first,root=tmp_path)
    table=(report/'tables/L2.tex').read_text()
    assert 'SDIRK2-mrSAV' in table and 'ETDRK4' not in table
    assert all(' & ' in line for line in table.splitlines() if line.startswith(('1 &','2 &','3 &')))
    assert json.loads((report/'analysis.json').read_text())['reference_passed']
    broken={**a,'trials':a['trials'][:-1]}
    path=tmp_path/'broken.json';path.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='Incomplete'):analyze_batch(path,root=tmp_path)
    with pytest.raises(ValueError):validate({**config,'k_levels':[1,2.5]})

import numpy as np
import pytest
from solver.integrate import integrate, step_sizes
from solver.mac.grid import MACGrid
from solver.mac_ns import MACNavierStokes
from solver.schemes.sdirk2 import SDIRK2
from experiments.problems import initial_velocity

def test_schedule_and_snapshot_times():
    grid=MACGrid(8,6)
    model=MACNavierStokes(grid,.1)
    state=model.state(0,initial_velocity(grid,.01))
    result=integrate(model,SDIRK2(),state,[.01,.02,.01],snapshots=[0,.015,.04])
    assert result.status=='complete'
    np.testing.assert_allclose(result.times,[0,.01,.03,.04])
    np.testing.assert_allclose(result.snapshot_times,[0,.01,.04])
    assert result.final.t==pytest.approx(.04)
    assert len(step_sizes(.1,dt=.03))==4
    with pytest.raises(ValueError): step_sizes(.1,steps=[.02,-.1])

def test_failed_step_retains_accepted_prefix():
    class FailSecond(SDIRK2):
        def step(self,model,state,dt):
            if state.t>0: raise RuntimeError('intentional failure')
            return super().step(model,state,dt)
    grid=MACGrid(5,4)
    model=MACNavierStokes(grid,.1)
    result=integrate(model,FailSecond(),model.state(0,initial_velocity(grid,.01)),[.01,.01])
    assert result.status=='failed'
    assert result.final.t==.01
    assert len(result.times)==2
    assert result.error=='RuntimeError: intentional failure'


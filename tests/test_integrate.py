import numpy as np
import pytest
from solver.integrate import integrate, step_sizes
from solver.schemes.imex_euler import IMEXEuler
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


def test_long_nearly_uniform_schedule_keeps_endpoint():
    steps = np.full(300000, 1e-4)
    steps[1000::1000] = np.nextafter(steps[1000::1000], np.inf)
    model = MACNavierStokes(MACGrid(3, 3), .1)
    initial = model.state(0., np.zeros(model.grid.size))
    model.force = lambda t: np.full(model.grid.size, np.nan)
    # A request at 30 must pass interval validation even though naive cumsum drifts.
    result = integrate(model, IMEXEuler(), initial, steps, snapshots=[30., 0., 0.])
    assert result.status == 'failed' and result.times == [0.]
    assert result.snapshot_requests == [0., 0.]
    assert result.snapshot_times == [0., 0.]
    with pytest.raises(ValueError, match='Snapshot time outside'):
        integrate(model, IMEXEuler(), initial, steps, snapshots=[30.+3e-12])


@pytest.mark.parametrize('initial_time', [0., .3])
def test_long_schedule_maps_midpoints_to_earlier_actual_nodes(initial_time):
    model = MACNavierStokes(MACGrid(3, 3), .1)
    initial = model.state(initial_time, np.zeros(model.grid.size))
    midpoint = initial_time+9.995
    requests = [initial_time+10., midpoint, np.nextafter(midpoint, np.inf),
                initial_time, midpoint]
    result = integrate(model, IMEXEuler(), initial, [.01]*1000, snapshots=requests)
    assert result.status == 'complete', result.error
    assert result.snapshot_requests == requests
    assert result.snapshot_times == [result.times[i] for i in (1000, 999, 999, 0, 999)]
    np.testing.assert_allclose(result.snapshot_times, [initial_time+t for t in (10., 9.99, 9.99, 0., 9.99)],
                               rtol=0., atol=3e-13)
    assert result.final.t == result.times[-1] == result.snapshots[0].t

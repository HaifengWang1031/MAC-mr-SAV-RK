"""Accepted-state diagnostics must not lose an already accepted snapshot on failure."""
import numpy as np
import pytest

from solver.adaptivity import IController, integrate_adaptive
from solver.integrate import Result, integrate
from solver.core import State
from solver.mac.grid import MACGrid
from solver.mac_ns import MACNavierStokes
from solver.schemes.registry import make_scheme


@pytest.mark.parametrize('adaptive', [False, True], ids=['fixed', 'adaptive'])
def test_observer_failure_retains_accepted_state_and_snapshot(adaptive):
    model = MACNavierStokes(MACGrid(3, 3), .1)
    initial = model.state(0., np.zeros(model.grid.size))
    observed = []

    def observe(state: State) -> None:
        observed.append(state.t)
        raise RuntimeError('observer stop')

    scheme = make_scheme('sdirk2')
    result: Result
    if adaptive:
        result = integrate_adaptive(model, scheme, initial, .02, .01,
                                    controller=IController(atol=1., rtol=0., max_step=.01,
                                                           estimator_order=2),
                                    snapshots=[.01], on_accept=observe)
    else:
        result = integrate(model, scheme, initial, [.01, .01], snapshots=[.01], on_accept=observe)
    assert result.status == 'failed' and 'observer stop' in result.error
    assert observed == [.01] and result.times == [0., .01]
    assert result.final.t == .01 and result.snapshot_times == [.01]
    assert len(result.diagnostics) == 2 and len(result.stages) == 1
    assert result.observer_cpu_seconds >= 0. and result.observer_wall_seconds >= 0.

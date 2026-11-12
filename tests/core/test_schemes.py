import numpy as np
import pytest
from solver.mac.grid import MACGrid
from solver.mac_ns import MACNavierStokes
from solver.core import State
from solver.schemes.sdirk2 import SDIRK2
from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV
from solver.schemes.sdirk3 import SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV

@pytest.mark.parametrize('scheme',[SDIRK2(),SDIRK2MRSAV(gamma=2.)])
def test_zero_velocity_and_scalar_decay(scheme):
    grid=MACGrid(6,4)
    model=MACNavierStokes(grid,nu=.1)
    u,v=grid.unpack(np.zeros(grid.size))
    state=State(0.,u,v,.3)
    trial=scheme.step(model,state,.03)
    assert np.linalg.norm(trial.state.u)<1e-14
    assert np.linalg.norm(trial.state.v)<1e-14
    if isinstance(scheme,SDIRK2MRSAV):
        eta=1-1/np.sqrt(2)
        expected=.3*(1-(1-2*eta)*2*.03)/(1+eta*2*.03)**2
        assert trial.state.r==pytest.approx(expected)
        assert max(s.scalar_residual for s in trial.stages)<1e-12



@pytest.mark.parametrize(
    ('scheme', 'expected_calls'),
    [
        (SDIRK2(), 1),
        (SDIRK2MRSAV(), 2),
        (SDIRK3(), 3),
        (SDIRK3MRSAV(), 4),
    ],
)
def test_each_stage_laplacian_is_reused(scheme, expected_calls):
    grid = MACGrid(6, 4)
    model = MACNavierStokes(grid, nu=0.1)
    original_apply_K = model.apply_K
    calls = 0

    def counted_apply_K(velocity):
        nonlocal calls
        calls += 1
        return original_apply_K(velocity)

    model.apply_K = counted_apply_K
    u, v = grid.unpack(np.zeros(grid.size))
    scheme.step(model, State(0.0, u, v, 0.3), 0.03)

    assert calls == expected_calls

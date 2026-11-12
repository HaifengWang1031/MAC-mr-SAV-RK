"""Explicit floor forcing, independently known trial values and retained hard failures."""
import numpy as np
import pytest

from solver.adaptivity import IController, PIController, ScalarErrorControl, integrate_adaptive
from solver.core import Trial
from tests.references.ode import DefectScheme, ODEModel, State


def test_force_floor_continues_the_accepted_pi_underflow_pattern():
    model = ODEModel()
    controller = PIController(atol=1e-3, rtol=0., min_step=.01, max_step=.1)
    scalar = ScalarErrorControl(.01**2/1.02/.5, 2.)
    initial = State(0., model.exact(0.))
    args = (model, DefectScheme(), initial, .5, .01)
    old = integrate_adaptive(*args, controller=controller, scalar_control=scalar)
    assert old.status == 'failed' and old.error == 'Step below min_step'
    assert old.attempts[-1]['accepted'] and old.attempts[-1]['error'] < 1.
    result = integrate_adaptive(*args, controller=controller, scalar_control=scalar,
                                min_step_policy='force')
    assert result.status == 'complete', result.error
    assert result.final.t == .5
    assert any(a['step'] == .01 for a in result.attempts[16:])
    np.testing.assert_allclose(result.final.velocity, model.exact(.5), atol=1e-14)


@pytest.mark.parametrize('component', ['velocity', 'scalar'])
def test_force_floor_records_finite_over_tolerance_steps_and_exact_terminal(component):
    model = ODEModel()
    controller = IController(atol=5e-7 if component == 'velocity' else 1e-3,
                             rtol=0., min_step=.01, max_step=.1)
    scalar = ScalarErrorControl(6e-5, 2.) if component == 'scalar' else None
    result = integrate_adaptive(model, DefectScheme(), State(0., model.exact(0.)), .035, .02,
                                controller=controller, scalar_control=scalar, min_step_policy='force')
    assert result.status == 'complete', result.error
    assert result.final.t == .035
    assert not result.attempts[0]['accepted']
    forced = [a for a in result.attempts if a['accepted'] and a['error'] > 1.]
    assert forced and all(a['step'] == .01 for a in forced)
    assert result.attempts[-1]['step'] < .01 and result.attempts[-1]['error'] <= 1.
    np.testing.assert_allclose(result.final.velocity, model.exact(.035), atol=1e-14)


@pytest.mark.parametrize('failure', ['nonfinite', 'stage'])
def test_force_floor_does_not_accept_nonfinite_or_failed_stage(failure):
    class BrokenScheme:
        def step(self, model, state, dt):
            if failure == 'stage':
                raise RuntimeError('stage residual failed')
            return Trial(State(state.t+dt, np.array([np.nan, 0.])), [], np.zeros(2))
    model = ODEModel()
    initial = State(0., model.exact(0.))
    result = integrate_adaptive(model, BrokenScheme(), initial, .1, .01,
                                controller=IController(min_step=.01), min_step_policy='force')
    assert result.status == 'failed' and result.final is initial
    assert not result.attempts[-1]['accepted']
    assert ('Nonfinite trial state' if failure == 'nonfinite' else 'stage residual failed') in result.error


def test_unknown_floor_policy_is_rejected():
    model = ODEModel()
    with pytest.raises(ValueError, match='min_step_policy'):
        integrate_adaptive(model, DefectScheme(), State(0., model.exact(0.)), .1, .01,
                           min_step_policy='unknown')

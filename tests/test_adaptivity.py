"""Embedded IMEX-SDIRK3(2) estimator and I/PI controller checks."""
import numpy as np
import pytest

from solver.adaptivity import IController, PIController, integrate_adaptive
from solver.schemes.sdirk3 import SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV
from tests.test_sdirk3 import ODEModel, State


@pytest.mark.parametrize('scheme', [SDIRK3(), SDIRK3MRSAV()])
def test_embedded_difference_has_cubic_local_scaling(scheme):
    model = ODEModel()
    initial = State(0., model.exact(0.))
    errors = []
    for h in (.2, .1, .05):
        trial = scheme.step(model, initial, h)
        assert trial.embedded_velocity is not None
        errors.append(np.linalg.norm(trial.state.velocity-trial.embedded_velocity))
    assert np.log2(errors[0]/errors[1]) > 2.6
    assert np.log2(errors[1]/errors[2]) > 2.7


@pytest.mark.parametrize('scheme', [SDIRK3(), SDIRK3MRSAV()])
def test_controller_rejects_and_commits_only_high_order_state(scheme):
    model = ODEModel()
    initial = State(0., model.exact(0.))
    control = IController(atol=1e-7, rtol=1e-5, max_step=.4)
    result = integrate_adaptive(model, scheme, initial, .5, .4,
                                controller=control, snapshots=[0., .25, .5])
    assert result.status == 'complete', result.error
    assert any(not attempt['accepted'] for attempt in result.attempts)
    assert len(result.times)-1 == sum(bool(item['accepted']) for item in result.attempts)
    assert all(item['error'] <= 1 for item in result.attempts if item['accepted'])
    assert result.final.t == pytest.approx(.5)
    assert np.linalg.norm(result.final.velocity-model.exact(.5)) < 1e-5
    assert len(result.snapshot_times) == 3
    assert result.snapshot_times[0] == 0.
    assert result.snapshot_times[-1] == pytest.approx(.5)
    assert all(len(stages) == 4 for stages in result.stages)


def test_rejected_trial_does_not_advance_state_on_failure():
    model = ODEModel()
    initial = State(0., model.exact(0.))
    control = IController(atol=1e-15, rtol=0., min_step=.1, max_rejections=1)
    result = integrate_adaptive(model, SDIRK3(), initial, .5, .4, controller=control)
    assert result.status == 'failed'
    assert result.final is initial
    assert result.times == [0.]
    assert all(not item['accepted'] for item in result.attempts)


def test_strict_snapshot_allows_short_alignment_step():
    from solver.schemes.sdirk2 import SDIRK2
    model = ODEModel()
    initial = State(0., model.exact(0.))
    result = integrate_adaptive(
        model, SDIRK2(), initial, .02, .01,
        controller=IController(atol=1., rtol=0., min_step=.005,
                               max_step=.01, estimator_order=2),
        snapshots=[0., .011, .02], strict_snapshots=True)
    assert result.status == 'complete', result.error
    np.testing.assert_allclose(result.snapshot_times, [0., .011, .02], atol=1e-14)
    assert any(item['step'] < .005 for item in result.attempts)


def test_sdirk2_embedded_pair_uses_order_two_controller():
    from solver.schemes.sdirk2 import SDIRK2
    from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV
    model = ODEModel()
    initial = State(0., model.exact(0.))
    for scheme in (SDIRK2(), SDIRK2MRSAV()):
        errors = []
        for step in (.2, .1, .05):
            trial = scheme.step(model, initial, step)
            assert trial.embedded_velocity is not None
            errors.append(np.linalg.norm(trial.state.velocity-trial.embedded_velocity))
        assert np.log2(errors[0]/errors[1]) > 1.7
        assert np.log2(errors[1]/errors[2]) > 1.8
        result = integrate_adaptive(model, scheme, initial, .1, .1,
                                    controller=IController(estimator_order=2))
        assert result.status == 'complete', result.error


@pytest.mark.parametrize('controller_class', [IController, PIController])
def test_small_mac_adaptive_trial_preserves_divergence(controller_class):
    from solver.mac.grid import MACGrid
    from solver.mac_ns import MACNavierStokes
    from experiments.problems import initial_velocity

    grid = MACGrid(8, 6)
    model = MACNavierStokes(grid, .1)
    initial = model.state(0., initial_velocity(grid, .01))
    for scheme in (SDIRK3(), SDIRK3MRSAV()):
        result = integrate_adaptive(model, scheme, initial, .03, .02,
                                    controller=controller_class(rtol=1e-3, max_step=.02))
        assert result.status == 'complete', result.error
        assert result.final.t == pytest.approx(.03)
        assert max(item['divergence_inf'] for item in result.diagnostics) < 1e-8


def test_pi_formula_first_step_and_zero_error():
    controller = PIController()
    error, old_error = .5, .25
    expected = .9*error**(-.7/3)*old_error**(.4/3)
    assert controller.accepted_factor(error, old_error) == pytest.approx(expected)
    assert controller.accepted_factor(error, None) == pytest.approx(
        IController().accepted_factor(error, None))
    assert np.isfinite(controller.accepted_factor(0., 0.))
    assert controller.accepted_factor(0., 0.) <= controller.max_factor
    with pytest.raises(ValueError, match='PI-controller'):
        PIController(proportional=.3, integral=.4)
    second_order = PIController(estimator_order=2)
    assert second_order.accepted_factor(error, old_error) == pytest.approx(
        .9*error**(-.7/2)*old_error**(.4/2))


@pytest.mark.parametrize('scheme', [SDIRK3(), SDIRK3MRSAV()])
def test_pi_rejection_uses_last_accepted_error(scheme):
    model = ODEModel()
    initial = State(0., model.exact(0.))
    controller = PIController(atol=1e-8, rtol=1e-6, max_step=.4)
    result = integrate_adaptive(model, scheme, initial, 1., .4, controller=controller)
    assert result.status == 'complete', result.error
    assert result.final.t == pytest.approx(1.)
    assert any(not item['accepted'] for item in result.attempts)
    previous = None
    for item in result.attempts:
        if previous is None:
            assert np.isnan(item['previous_accepted_error'])
        else:
            assert item['previous_accepted_error'] == previous
        if item['accepted']:
            assert item['factor'] == pytest.approx(
                controller.accepted_factor(item['error'], previous))
            previous = item['error']
        else:
            assert item['factor'] < 1.
    assert np.linalg.norm(result.final.velocity-model.exact(1.)) < 1e-5

import numpy as np
import pytest

from experiments.periodic_ns.problems import Manufactured, SCHEMES, scheme, smooth_stokes
from solver.adaptivity import IController, PIController, ScalarErrorControl, integrate_adaptive
from solver.fourier import FourierModel, FourierSpace
from solver.integrate import integrate
from solver.schemes.registry import DEFAULT_SCHEMES, scheme_info


def setup():
    s = FourierSpace(21, 18)
    ref = Manufactured(s, .1)
    model = FourierModel(s, .1, force=lambda t: s.to_spectral(ref.force(t)))
    initial = model.state(0., s.to_spectral(ref.exact(0.)))
    return model, initial, ref


@pytest.mark.parametrize('name', SCHEMES)
def test_nonlinear_mms_velocity_order_and_stage_equations(name):
    model, initial, ref = setup()
    before = initial.u.copy(), initial.v.copy()
    errors = []
    for n in (10, 20, 40):
        result = integrate(model, scheme(name), initial, [.4/n]*n)
        assert result.status == 'complete', result.error
        error = np.stack([result.final.u, result.final.v])-ref.exact(result.final.t)
        errors.append(np.sqrt(model.space.area*np.mean(np.sum(error**2, axis=0))))
        assert max(d['divergence_inf'] for d in result.diagnostics) < 1e-12
        assert max(s['residual'] for step in result.stages for s in step) < 1e-12
        assert result.final_stages[0].pressure.shape == model.space.shape
    target = scheme_info(name).order
    assert min(np.log2(np.array(errors[:-1])/errors[1:])) > target-.25, (name, errors)
    np.testing.assert_array_equal(initial.u, before[0])
    np.testing.assert_array_equal(initial.v, before[1])
    variable = integrate(model, scheme(name, 1000.), initial, [.01, .02, .01])
    assert variable.status == 'complete', variable.error


@pytest.mark.parametrize('name', DEFAULT_SCHEMES)
@pytest.mark.parametrize('controller_type', [IController, PIController])
def test_adaptive_rejection_and_independent_physical_norm(name, controller_type):
    model, initial, ref = setup()
    c = controller_type(atol=1e-8, rtol=2e-5, estimator_order=2 if name.startswith('sdirk2') else 3)
    states = []
    options = dict(controller=c, snapshots=[0., .13, .3])
    first = integrate_adaptive(model, scheme(name), initial, .3, .25, on_accept=states.append, **options)
    # An independently evaluated physical norm must give the same decisions.
    norm = lambda z: float(np.sqrt(model.space.area*np.mean(np.sum(model.space.to_physical(z)**2, axis=0))))
    second = integrate_adaptive(model, scheme(name), initial, .3, .25, error_norm=norm, **options)
    assert first.status == second.status == 'complete', first.error
    assert any(not a['accepted'] for a in first.attempts)
    assert first.final.t == states[-1].t == first.times[-1] == .3
    np.testing.assert_allclose(first.times, second.times, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose([a['error'] for a in first.attempts],
                               [a['error'] for a in second.attempts], rtol=1e-8, atol=1e-10)
    error = np.stack([first.final.u, first.final.v])-ref.exact(.3)
    assert np.sqrt(model.space.area*np.mean(np.sum(error**2, axis=0))) < .002
    assert len(states) == len(first.times)-1


def test_mean_evolution_small_terminal_step_and_joint_scalar_control():
    s = FourierSpace(12, 10)
    force = s.velocity(np.ones(s.shape), np.full(s.shape, 2.))
    model = FourierModel(s, .1, force=lambda t: force)
    initial = model.state(0., s.velocity(np.full(s.shape, .3), np.full(s.shape, -.2)))
    for name in DEFAULT_SCHEMES:
        result = integrate_adaptive(model, scheme(name), initial, .021, .02,
                                    controller=IController(atol=1e-4, rtol=0., min_step=.01,
                                                           max_step=.02, estimator_order=2 if name.startswith('sdirk2') else 3))
        assert result.status == 'complete', result.error
        assert result.final.t == .021 and np.diff(result.times)[-1] < .01
        np.testing.assert_allclose(result.final.u, .321, atol=1e-14)
        np.testing.assert_allclose(result.final.v, -.158, atol=1e-14)
    model, initial, _ = setup()
    result = integrate_adaptive(model, scheme('sdirk3_mrsav', 1000.), initial, .05, .01,
                                controller=PIController(atol=1e-8, rtol=1e-4),
                                scalar_control=ScalarErrorControl(atol=1e-6, gamma=1000.))
    assert result.status == 'complete', result.error
    assert all(a['error'] == max(a['error_u'], a['error_r']) for a in result.attempts)
    assert all(a['error'] <= 1 for a in result.attempts if a['accepted'])


def test_infinite_modal_spatial_convergence_without_temporal_error():
    errors = []
    for n in (12, 18, 24):
        s = FourierSpace(n, n)
        exact, force = smooth_stokes(s)
        model = FourierModel(s, 1.)
        solved = model.solve(s.to_spectral(force), mass=0., viscosity=1.)
        difference = s.to_physical(solved.velocity)-exact
        errors.append(np.sqrt(s.area*np.mean(np.sum(difference**2, axis=0))))
    assert errors[1] < errors[0]/50 and errors[2] < errors[1]/100, errors


@pytest.mark.parametrize('name', SCHEMES)
def test_gradient_force_reaches_stage_pressure_without_driving_velocity(name):
    from solver.schemes.sdirk2 import ETA
    s = FourierSpace(18, 15)
    p = s.to_spectral(np.cos(s.x)*np.sin(s.y))
    model = FourierModel(s, .1, force=lambda t: s.gradient(p))
    trial = scheme(name).step(model, model.state(0., model.zero_velocity()), .01)
    diagonal = 1. if scheme_info(name).order == 1 else ETA if name.startswith('sdirk2') else .25
    np.testing.assert_allclose(trial.state.u, 0., atol=1e-14)
    np.testing.assert_allclose(trial.state.v, 0., atol=1e-14)
    np.testing.assert_allclose(trial.stages[0].pressure, diagonal*np.cos(s.x)*np.sin(s.y), atol=1e-14)

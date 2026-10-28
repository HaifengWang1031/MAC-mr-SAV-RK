"""Independent one-step and convergence references for first-order IMEX methods."""
import numpy as np
import pytest

from solver.schemes.imex_euler import IMEXEuler
from solver.schemes.imex_euler_mrsav import IMEXEulerMRSAV
from solver.integrate import integrate
from tests.test_sdirk3 import ODEModel, State


def test_euler_uses_old_force_and_convection_and_backward_euler_diffusion():
    model = ODEModel()
    initial = State(.2, np.array([.4, -.2]))
    original = initial.velocity.copy()
    h = .03
    rhs = original+h*(model.force(initial.t)-model.nonlinear(original))
    expected = np.linalg.solve(np.eye(2)+h*model.L, rhs)
    trial = IMEXEuler().step(model, initial, h)
    np.testing.assert_allclose(trial.state.velocity, expected, atol=1e-15)
    np.testing.assert_array_equal(initial.velocity, original)
    assert trial.state.t == initial.t+h
    assert len(trial.stages) == 1 and trial.embedded_velocity is None
    assert trial.stages[0].residual < 1e-14


@pytest.mark.parametrize('dt', [0., -1., float('nan'), float('inf')])
def test_euler_rejects_invalid_steps(dt):
    model = ODEModel()
    with pytest.raises(ValueError, match='Invalid step'):
        IMEXEuler().step(model, State(0., model.exact(0.)), dt)


@pytest.mark.parametrize('gamma', [0., 1., 1000.])
def test_sav_matches_independent_coupled_linear_system_and_energy_balance(gamma):
    model = ODEModel()
    initial = State(.2, np.array([.4, -.2]), .3)
    old = initial.velocity.copy()
    h = .03
    B, f = model.nonlinear(old), model.force(initial.t)
    block = np.block([[np.eye(2)+h*model.L, -h*B[:, None]],
                      [h*B[None, :], np.array([[1+gamma*h]])]])
    expected = np.linalg.solve(block, np.r_[old+h*f-h*B, initial.r])
    trial = IMEXEulerMRSAV(gamma).step(model, initial, h)
    u, r = trial.state.velocity, trial.state.r
    np.testing.assert_allclose(np.r_[u, r], expected, atol=1e-15)
    np.testing.assert_array_equal(initial.velocity, old)
    stage = trial.stages[0]
    assert len(trial.stages) == 1 and stage.candidates == [r]
    assert stage.residual < 1e-14 and stage.scalar_residual < 1e-14
    assert trial.embedded_velocity is None
    energy_change = .5*(u@u-old@old+(r-1)**2-(initial.r-1)**2)
    balance = energy_change+.5*((u-old)@(u-old)+(r-initial.r)**2) \
        +h*(u@model.L@u+gamma*r*(r-1)-f@u)
    assert abs(balance) < 2e-15


def test_pure_diffusion_and_zero_convection_scalar_decay():
    class Decay(ODEModel):
        def nonlinear(self, velocity): return np.zeros(2)
        def force(self, t): return np.zeros(2)
    model = Decay()
    initial = State(0., np.array([.4, -.2]), .3)
    h = .03
    expected = initial.velocity/(1+h*np.diag(model.L))
    for scheme in (IMEXEuler(), IMEXEulerMRSAV(gamma=2.)):
        trial = scheme.step(model, initial, h)
        np.testing.assert_allclose(trial.state.velocity, expected, atol=1e-15)
        assert trial.state.r == pytest.approx(.3/(1+2*h) if isinstance(scheme, IMEXEulerMRSAV) else 0.)


@pytest.mark.parametrize('scheme', [IMEXEuler(), IMEXEulerMRSAV(1.), IMEXEulerMRSAV(1000.)])
def test_first_order_on_independent_nonlinear_manufactured_ode(scheme):
    model = ODEModel()
    errors, scalars = [], []
    for count in (400, 800, 1600):
        result = integrate(model, scheme, State(0., model.exact(0.)), [2/count]*count)
        assert result.status == 'complete', result.error
        errors.append(np.linalg.norm(result.final.velocity-model.exact(2.)))
        scalars.append(abs(result.final.r))
    assert min(np.log2(np.array(errors[:-1])/errors[1:])) > .9, errors
    if isinstance(scheme, IMEXEulerMRSAV):
        assert min(np.log2(np.array(scalars[:-1])/scalars[1:])) > .85, scalars


@pytest.mark.parametrize('gamma', [-1., float('nan'), float('inf')])
def test_sav_rejects_invalid_damping(gamma):
    with pytest.raises(ValueError, match='gamma'):
        IMEXEulerMRSAV(gamma)

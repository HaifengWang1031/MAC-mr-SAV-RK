"""Joint velocity/scalar control, including independent component histories."""
from dataclasses import replace

import numpy as np
import pytest

from solver.adaptivity import IController, PIController, ScalarErrorControl, integrate_adaptive
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV
from tests.references.ode import DefectScheme, ODEModel, State


@pytest.mark.parametrize('controller_class', [IController, PIController])
def test_joint_acceptance_orders_and_pi_history(controller_class):
    model = ODEModel()
    controller = controller_class(atol=1e-3, rtol=0., max_step=.2)
    scalar = ScalarErrorControl(atol=1e-3, gamma=2.)
    result = integrate_adaptive(model, DefectScheme(), State(0., model.exact(0.)),
                                .25, .2, controller=controller, scalar_control=scalar)
    assert result.status == 'complete', result.error
    assert any(not a['accepted'] for a in result.attempts)
    old_u = old_r = None
    r = 0.
    for a in result.attempts:
        h = a['step']
        defect = (r+h*h)/(1+2*h)-np.exp(-2*h)*r
        assert a['scalar_defect'] == pytest.approx(defect)
        assert a['error_r'] == pytest.approx(abs(defect)/scalar.atol)
        assert a['error'] == max(a['error_u'], a['error_r'])
        assert a['accepted'] == (a['error'] <= 1)
        if old_r is None:
            assert np.isnan(a['previous_accepted_error_r'])
        else:
            assert a['previous_accepted_error_u'] == old_u
            assert a['previous_accepted_error_r'] == old_r
        if a['accepted']:
            expected = min(controller.accepted_factor(a['error_u'], old_u),
                           replace(controller, estimator_order=2).accepted_factor(
                               a['error_r'], old_r))
            old_u, old_r = a['error_u'], a['error_r']
            r = (r+h*h)/(1+2*h)
        else:
            expected = min(.8, IController.accepted_factor(controller, a['error_u'], None),
                           IController.accepted_factor(replace(controller, estimator_order=2),
                                                       a['error_r'], None))
        assert a['factor'] == pytest.approx(expected)
    np.testing.assert_allclose(result.final.velocity, model.exact(.25), atol=1e-14)
    assert result.final.r == pytest.approx(r)


def test_scalar_controls_local_defect_not_propagated_old_error():
    model = ODEModel()
    initial = State(0., model.exact(0.), .3)
    scalar = ScalarErrorControl(atol=.01, gamma=2.)
    result = integrate_adaptive(model, DefectScheme(), initial, .01, .01,
                                controller=IController(atol=1., rtol=0.),
                                scalar_control=scalar)
    assert result.status == 'complete'
    assert result.attempts[0]['accepted']
    assert abs(result.final.r) > scalar.atol
    assert result.attempts[0]['scalar_defect'] == pytest.approx(
        result.final.r-np.exp(-.02)*initial.r)


def test_mrsav_scalar_local_defect_has_quadratic_scaling():
    model = ODEModel()
    initial = State(0., model.exact(0.))
    scalar = ScalarErrorControl(atol=1., gamma=1.)
    errors = [abs(scalar.defect(initial.r, SDIRK3MRSAV().step(model, initial, h).state.r, h))
              for h in (.02, .01, .005)]
    assert min(np.log2(np.asarray(errors[:-1])/errors[1:])) > 1.9


@pytest.mark.parametrize('atol,gamma', [(0., 1.), (-1., 1.), (1., -1.), (np.inf, 1.)])
def test_invalid_scalar_controls(atol, gamma):
    with pytest.raises(ValueError, match='scalar error controls'):
        ScalarErrorControl(atol=atol, gamma=gamma)

import numpy as np
import pytest

from tests.references.channel import DiscreteManufactured
from solver.schemes.registry import SCHEMES, make_scheme as scheme
from solver.adaptivity import IController, PIController, ScalarErrorControl, integrate_adaptive
from solver.channel import ChannelGrid, ChannelModel
from solver.integrate import integrate
from solver.schemes.registry import DEFAULT_SCHEMES, scheme_info


def setup():
    grid = ChannelGrid(12, 10, 2*np.pi, 1.)
    ref = DiscreteManufactured(grid, .1)
    model = ChannelModel(grid, .1, force=ref.force)
    return model, model.state(0., ref.exact(0.)), ref


@pytest.mark.parametrize('name', SCHEMES)
@pytest.mark.parametrize('gamma', [1., 1000.])
def test_nonlinear_fixed_grid_orders_full_stages_and_variable_steps(name, gamma):
    model, initial, ref = setup()
    old = initial.u.copy(), initial.v.copy()
    errors = []
    for n in [40, 80, 160]:
        result = integrate(model, scheme(name, gamma), initial, [.4/n]*n)
        assert result.status == 'complete', result.error
        errors.append(model.grid.norm(model.vector(result.final)-ref.exact(result.final.t)))
        assert max(d['divergence_inf'] for d in result.diagnostics) < 1e-11
        assert max(s['residual'] for step in result.stages for s in step) < 1e-11
        assert max(s['scalar_residual'] for step in result.stages for s in step) < 1e-11
        assert result.final_stages[0].pressure.shape == (10, 12)
    target = scheme_info(name).order
    assert min(np.log2(np.array(errors[:-1])/errors[1:])) > target-.1, (name, gamma, errors)
    variable = integrate(model, scheme(name, gamma), initial, [.01, .02, .01])
    assert variable.status == 'complete', variable.error
    np.testing.assert_array_equal(initial.u, old[0]); np.testing.assert_array_equal(initial.v, old[1])


@pytest.mark.parametrize('name', DEFAULT_SCHEMES)
@pytest.mark.parametrize('controller_type', [IController, PIController])
def test_adaptive_rejection_physical_norm_callbacks_and_exact_endpoint(name, controller_type):
    model, initial, ref = setup()
    controller = controller_type(atol=1e-9, rtol=1e-6,
                                 estimator_order=2 if name.startswith('sdirk2') else 3)
    states = []
    options = dict(controller=controller, snapshots=[0., .13, .3])
    a = integrate_adaptive(model, scheme(name), initial, .3, .25, on_accept=states.append, **options)
    def norm(w):
        u, v = model.grid.unpack(w)
        return float(np.sqrt(model.grid.area*(np.sum(u[:, :-1]**2)+np.sum(v[1:-1]**2))))
    b = integrate_adaptive(model, scheme(name), initial, .3, .25, error_norm=norm, **options)
    assert a.status == b.status == 'complete', a.error
    assert any(not trial['accepted'] for trial in a.attempts)
    assert a.final.t == a.times[-1] == states[-1].t == a.snapshot_times[-1] == .3
    np.testing.assert_allclose(a.times, b.times, rtol=1e-7, atol=1e-10)
    assert model.grid.norm(model.vector(a.final)-ref.exact(.3)) < 1e-4


def test_sub_min_step_endpoint_and_joint_scalar_control():
    model, initial, _ = setup()
    result = integrate_adaptive(model, scheme('sdirk3_mrsav', 1000.), initial, .0101, .01,
                                controller=IController(atol=1., rtol=0., min_step=.01, max_step=.01),
                                scalar_control=ScalarErrorControl(1., 1000.), snapshots=[.0101])
    assert result.status == 'complete', result.error
    assert result.final.t == result.snapshot_times[-1] == .0101
    assert result.times[-1]-result.times[-2] < .01
    assert all(a['error'] == max(a['error_u'], a['error_r']) for a in result.attempts)


def test_continuous_nonlinear_mms_spatial_order():
    from tests.references.channel import Manufactured
    errors = []
    for ny in [6, 12, 24]:
        grid = ChannelGrid(2*ny, ny, 2*np.pi, 1.)
        ref = Manufactured(grid, .1)
        model = ChannelModel(grid, .1, force=ref.force)
        initial = model.state(0., ref.amplitude(0.)*ref.initial_shape)
        result = integrate(model, scheme('sdirk3'), initial, [.001]*100)
        assert result.status == 'complete', result.error
        errors.append(grid.norm(model.vector(result.final)-ref.exact(result.final.t)))
    assert min(np.log2(np.array(errors[:-1])/errors[1:])) > 1.8, errors

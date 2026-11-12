"""Natural-node continuation is owned by the solver, not research persistence."""
import numpy as np
import pytest

from solver.adaptivity import PIController, ScalarErrorControl, integrate_adaptive
from solver.schemes.registry import make_scheme
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV
from tests.references.ode import ODEModel, State


def test_bounded_history_and_joint_pi_continuation_keep_the_natural_trajectory():
    model = ODEModel()
    initial = State(0., model.exact(0.))
    controller = PIController(atol=1e-8, rtol=1e-5, max_step=.015,
                              proportional=.8, integral=.3)
    options = {'controller': controller, 'scalar_control': ScalarErrorControl(1e-3, 1000.)}
    full = integrate_adaptive(model, SDIRK3MRSAV(1000.), initial, .4, .001, **options)

    def stop_at_natural_node(result):
        if result.accepted_steps == 5:
            raise RuntimeError('intentional checkpoint boundary')

    prefix = integrate_adaptive(model, SDIRK3MRSAV(1000.), initial, .4, .001,
                                retain_history=False, on_advance=stop_at_natural_node, **options)
    assert prefix.status == 'failed' and prefix.accepted_steps == 5
    assert prefix.continuation.previous_u is not None
    assert prefix.continuation.previous_r is not None
    tail = integrate_adaptive(model, SDIRK3MRSAV(1000.), prefix.final, .4-prefix.final.t, .001,
                              retain_history=False, continuation=prefix.continuation, **options)
    assert full.status == tail.status == 'complete'
    assert len(tail.times) == 2 and len(tail.stages) == len(tail.attempts) == 1
    assert tail.continuation.accepted_steps == full.accepted_steps
    assert tail.continuation.attempted_steps == full.attempted_steps
    np.testing.assert_allclose(tail.final.velocity, full.final.velocity, rtol=0., atol=1e-13)
    assert abs(tail.final.r-full.final.r) < 1e-13
    assert np.linalg.norm(full.final.velocity-model.exact(.4)) < 1e-3


@pytest.mark.parametrize('retain_history', [True, False])
@pytest.mark.parametrize('observer', ['on_accept', 'on_advance'])
def test_failed_accepted_observer_returns_matching_joint_pi_continuation(retain_history, observer):
    model = ODEModel()
    initial = State(0., model.exact(0.))
    controller = PIController(atol=1e-8, rtol=1e-5, max_step=.015,
                              proportional=.8, integral=.3)
    options = {'controller': controller, 'scalar_control': ScalarErrorControl(1e-3, 1000.)}
    full_nodes = []
    full = integrate_adaptive(model, make_scheme('sdirk3_mrsav', 1000.), initial, .4, .001,
                              on_accept=lambda state: full_nodes.append(state.t), **options)
    assert full.status == 'complete', full.error
    prefix_nodes = []

    def stop_at_fifth_node(value):
        state = value.final if observer == 'on_advance' else value
        prefix_nodes.append(state.t)
        if len(prefix_nodes) == 5:
            raise RuntimeError('accepted observer stop')

    prefix = integrate_adaptive(model, make_scheme('sdirk3_mrsav', 1000.), initial, .4, .001,
                                retain_history=retain_history,
                                **{observer: stop_at_fifth_node}, **options)
    assert prefix.status == 'failed' and 'accepted observer stop' in prefix.error
    assert prefix.accepted_steps == 5 and prefix.final.t == prefix_nodes[-1]
    context = prefix.continuation
    assert context is not None
    assert context.accepted_steps == prefix.accepted_steps
    assert context.attempted_steps == prefix.attempted_steps
    assert context.previous_u is not None and context.previous_r is not None
    assert context.previous_error == prefix.attempts[-1]['error']
    assert context.previous_u == prefix.attempts[-1]['error_u']
    assert context.previous_r == prefix.attempts[-1]['error_r']

    tail_nodes = []
    tail = integrate_adaptive(model, make_scheme('sdirk3_mrsav', 1000.), prefix.final,
                              .4-prefix.final.t, .001, controller=controller,
                              scalar_control=options['scalar_control'],
                              retain_history=retain_history, continuation=context,
                              on_accept=lambda state: tail_nodes.append(state.t))
    assert tail.status == 'complete', tail.error
    np.testing.assert_array_equal(prefix_nodes+tail_nodes, full_nodes)
    assert tail.continuation.accepted_steps == prefix.accepted_steps+tail.accepted_steps == full.accepted_steps
    assert tail.continuation.attempted_steps == prefix.attempted_steps+tail.attempted_steps == full.attempted_steps
    np.testing.assert_allclose(tail.final.velocity, full.final.velocity, rtol=0., atol=1e-13)
    assert abs(tail.final.r-full.final.r) < 1e-13
    assert np.linalg.norm(tail.final.velocity-model.exact(.4)) < 1e-3

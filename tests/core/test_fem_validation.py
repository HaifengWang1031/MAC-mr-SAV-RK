"""Six shared formats at the FEM Model/step/integrate seam, not a PDE order proof."""
from math import fsum

import numpy as np
import pytest

from solver.fem import FEMModel, Space
from solver.fem.space import Array
from solver.integrate import integrate, step_sizes
from solver.schemes.registry import SCHEMES, make_scheme, scheme_info


def exponential_trajectory() -> tuple[FEMModel, Array]:
    """Exact semidiscrete u(t)=exp(t)w, r(t)=0; no time-scheme reference."""
    model = FEMModel(Space(4, 3, lx=2., ly=1.), nu=.07)
    projected = model.project(lambda x, y: (np.sin(2*x)*np.sin(3*y),
                                           np.sin(x)*np.sin(2*y)))
    velocity = model.solve(projected, mass=1., viscosity=0.).velocity
    velocity /= np.sqrt(model.inner(velocity, velocity))
    diffusion = model.nu*model.apply_K(velocity)
    convection = model.nonlinear(velocity)
    assert model.max_abs(model.apply_D(velocity)) < 1e-12
    assert abs(model.inner(convection, velocity)) < 1e-12
    assert model.inner(convection, convection) > .1
    # Spatial operators deliberately define this finite-dimensional forcing.
    # Their independent spatial accuracy/weak-form checks live in test_fem_model.py.
    model.force = lambda t: np.exp(t)*(velocity+diffusion)+np.exp(2*t)*convection
    return model, velocity


@pytest.mark.parametrize('name', SCHEMES)
def test_exponential_semidiscrete_temporal_refinement(name: str) -> None:
    model, velocity = exponential_trajectory()
    scheme = make_scheme(name, gamma=1.)
    errors = []
    scalar_errors = []
    for count in (16, 32, 64):
        result = integrate(model, scheme, model.state(0., velocity), [.4/count]*count)
        assert result.status == 'complete', result.error
        difference = model.vector(result.final)-np.exp(.4)*velocity
        errors.append(np.sqrt(model.inner(difference, difference)))
        scalar_errors.append(abs(result.final.r))
    rates = np.log2(np.array(errors[:-1])/errors[1:])
    if not name.endswith('mrsav'):
        order = scheme_info(name).order
        assert np.all((order-.3 < rates) & (rates < order+.3)), (errors, rates)
    else:
        assert errors[-1] < errors[0]/2, errors
        assert scalar_errors[-1] < scalar_errors[0], scalar_errors
    # Frozen initial data cannot pass even the loosest first-order error gate.
    frozen_error = (np.exp(.4)-1)*np.sqrt(model.inner(velocity, velocity))
    assert errors[-1] < .01 < frozen_error


@pytest.mark.parametrize('name', SCHEMES)
@pytest.mark.parametrize('mode', ['fixed', 'prescribed'])
def test_six_formats_record_actual_nodes_residuals_roots_and_preserve_input(name: str, mode: str) -> None:
    model, velocity = exponential_trajectory()
    scheme = make_scheme(name, gamma=2.)
    initial = model.state(.3, velocity, r=.17)
    original_u, original_v = initial.u.copy(), initial.v.copy()
    steps = (step_sizes(.007, dt=.002) if mode == 'fixed'
             else step_sizes(.007, steps=[.001, .003, .002, .001]))
    nodes = [.3+fsum(steps[:k]) for k in range(len(steps)+1)]
    trial = scheme.step(model, initial, float(steps[0]))
    assert trial.state.t == initial.t+steps[0]
    assert len(trial.stages) == scheme_info(name).n_stages
    np.testing.assert_array_equal(initial.u, original_u)
    np.testing.assert_array_equal(initial.v, original_v)
    assert model.max_abs(model.apply_D(model.vector(trial.state))) < 1e-12
    progress_times: list[float] = []
    result = integrate(model, scheme, initial, steps, snapshots=nodes,
                       progress=lambda k, t, seconds: progress_times.append(t))
    assert result.status == 'complete', result.error
    np.testing.assert_allclose(result.times, nodes, rtol=0., atol=2e-16)
    np.testing.assert_array_equal(result.times[1:], progress_times)
    np.testing.assert_array_equal(result.snapshot_times, result.times)
    assert result.final.t == result.times[-1] == result.snapshots[-1].t
    assert len(result.diagnostics) == len(nodes)
    assert len(result.stages) == len(steps)
    np.testing.assert_array_equal(initial.u, original_u)
    np.testing.assert_array_equal(initial.v, original_v)
    assert initial.t == .3 and initial.r == .17
    sav = name.endswith('mrsav')
    assert result.final.r == result.final_stages[-1].r
    if sav:
        assert abs(result.final.r) > .01
    else:
        assert result.final.r == 0.
    for pressure_stage in trial.stages+result.final_stages:
        assert np.isfinite(pressure_stage.pressure).all()
        assert abs(model.pressure_integral(pressure_stage.pressure)) < 1e-11
    for diagnostic, snapshot in zip(result.diagnostics, result.snapshots):
        assert all(np.isfinite(value) for value in diagnostic.values())
        assert diagnostic['r'] == snapshot.r
        assert model.max_abs(model.apply_D(model.vector(snapshot))) < 1e-12
        assert diagnostic['continuity_residual'] < 1e-12
        assert diagnostic['divergence_l2'] > 1e-3
    for stages in result.stages:
        assert len(stages) == scheme_info(name).n_stages
        for stage in stages:
            assert 0. <= stage['residual'] < 1e-10
            assert 0. <= stage['continuity_residual'] < 1e-12
            assert 0. <= stage['scalar_residual'] < 1e-12
            assert np.isfinite(stage['divergence_inf']) and stage['divergence_inf'] > 1e-3
            candidates = stage['candidates']
            assert stage['root_count'] == len(candidates) == len(stage['root_residuals'])
            if sav:
                assert candidates == stage['root_residuals'] == []
                assert stage['correction_count'] == scheme_info(name).n_corrections
                assert np.isfinite(stage['corrections']).all()
                assert stage['r'] == stage['corrections'][-1]
            else:
                assert candidates == [] and stage['r'] == 0.


@pytest.mark.parametrize('name', SCHEMES)
def test_six_formats_reject_invalid_steps_without_mutation(name: str) -> None:
    model, velocity = exponential_trajectory()
    initial = model.state(.3, velocity, r=.17)
    original_u, original_v = initial.u.copy(), initial.v.copy()
    scheme = make_scheme(name)
    for dt in (0., -.001, np.nan, np.inf):
        with pytest.raises(ValueError):
            scheme.step(model, initial, dt)
        with pytest.raises(ValueError, match='Invalid step sequence'):
            integrate(model, scheme, initial, [.001, dt])
    for steps in (np.array([]), np.array([[.001, .002]])):
        with pytest.raises(ValueError, match='Invalid step sequence'):
            integrate(model, scheme, initial, steps)
    np.testing.assert_array_equal(initial.u, original_u)
    np.testing.assert_array_equal(initial.v, original_v)
    assert initial.t == .3 and initial.r == .17


@pytest.mark.parametrize('name', SCHEMES)
@pytest.mark.parametrize('failure', ['nonfinite_force', 'linear_solve'])
def test_six_formats_preserve_only_accepted_prefix_on_real_failure(name: str, failure: str) -> None:
    model = FEMModel(Space(4, 3, lx=2., ly=1.), nu=.07,
                     tolerance=1e-30 if failure == 'linear_solve' else 1e-9)
    force = model.project(lambda x, y: (np.sin(2*x+y), np.cos(x-3*y)))
    bad_force = force if failure == 'linear_solve' else np.full_like(force, np.nan)
    model.force = lambda t: model.zero_velocity() if t <= .001 else bad_force
    initial = model.state(0., model.zero_velocity())
    scheme = make_scheme(name)
    progress_times: list[float] = []
    result = integrate(model, scheme, initial, [.001, .002, .001], snapshots=[0., .001, .003, .004],
                       progress=lambda k, t, seconds: progress_times.append(t))
    # Euler samples only the old force; SDIRK also sees nodes inside the failing step.
    accepted_nodes = [0., .001, .003] if scheme_info(name).order == 1 else [0., .001]
    assert result.status == 'failed'
    assert ('RuntimeError' in result.error and 'residual' in result.error
            if failure == 'linear_solve' else 'ValueError' in result.error)
    assert result.times == accepted_nodes
    assert result.final.t == accepted_nodes[-1]
    assert len(result.stages) == len(accepted_nodes)-1
    assert len(result.diagnostics) == len(accepted_nodes)
    assert result.snapshot_times == accepted_nodes
    assert progress_times == accepted_nodes[1:]
    np.testing.assert_array_equal(model.vector(result.final), 0.)
    np.testing.assert_array_equal(model.vector(initial), 0.)
    assert result.final.r == initial.r == 0.
    assert len(result.final_stages) == scheme_info(name).n_stages
    # Direct public step propagates the error instead of returning an accepted trial.
    with pytest.raises(RuntimeError if failure == 'linear_solve' else ValueError):
        scheme.step(model, model.state(.004, model.zero_velocity()), .001)


def test_temporal_oracle_discriminates_missing_force_and_lower_order() -> None:
    model, velocity = exponential_trajectory()
    initial = model.state(0., velocity)
    errors = []
    for name in ('imex_euler', 'sdirk3'):
        result = integrate(model, make_scheme(name), initial, [.4/64]*64)
        assert result.status == 'complete', result.error
        difference = model.vector(result.final)-np.exp(.4)*velocity
        errors.append(np.sqrt(model.inner(difference, difference)))
    assert errors[1] < 2e-6 < 1e-3 < errors[0]
    # A physically distinct unforced trajectory is a negative control, not a mock.
    model.force = lambda t: model.zero_velocity()
    unforced = integrate(model, make_scheme('sdirk3'), initial, [.4/64]*64)
    assert unforced.status == 'complete', unforced.error
    difference = model.vector(unforced.final)-np.exp(.4)*velocity
    assert np.sqrt(model.inner(difference, difference)) > .1

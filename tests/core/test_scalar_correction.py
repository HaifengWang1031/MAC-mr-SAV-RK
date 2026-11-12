"""Independent coupled linear systems check fixed-count scalar stage updates."""
import numpy as np
import pytest

from solver.schemes.scalar_correction import solve_scalar_stage
from solver.schemes.registry import make_scheme, scheme_info
from tests.references.ode import ODEModel, State


@pytest.mark.parametrize(('order', 'count'), [(1, 1), (2, 1), (3, 2)])
def test_frozen_updates_match_independent_coupled_block_system(order, count):
    L = np.array([[1.8, .2], [.2, 1.4]])
    B, rhs_u = np.array([.7, -.4]), np.array([.6, -.2])
    dt, d, rhs_r, predictor = .2, 1.7, .23, -.25
    U, V = np.linalg.solve(L, rhs_u), np.linalg.solve(L, B)
    result = solve_scalar_stage(predictor, dt, d, rhs_r, B@U, B@V, order=order)
    previous = predictor
    expected_values = []
    for _ in range(count):
        q = sum(previous**j for j in range(order))
        block = np.block([[L, -dt*q*B[:, None]], [dt*q*B[None, :], np.array([[d]])]])
        expected = np.linalg.solve(block, np.r_[rhs_u-dt*q*B, rhs_r])
        previous = float(expected[-1])
        expected_values.append(previous)
    assert len(result.values) == count
    np.testing.assert_allclose(result.values, expected_values, atol=2e-16, rtol=0.)
    np.testing.assert_allclose(U-dt*result.coupling*V, expected[:-1], atol=2e-16, rtol=0.)
    assert result.factor == pytest.approx(q)


@pytest.mark.parametrize('name', ['imex_euler_mrsav', 'sdirk2_mrsav', 'sdirk3_mrsav'])
def test_stage_counts_predictors_and_two_stokes_columns_without_root_finding(name, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Fixed corrections must not call polynomial root finding')
    monkeypatch.setattr(np, 'roots', forbidden)
    monkeypatch.setattr(np.polynomial.polynomial, 'polyroots', forbidden)
    model = ODEModel()
    original = model.solve_columns
    calls = []

    def columns(rhs, *, mass, viscosity):
        calls.append(len(rhs))
        return original(rhs, mass=mass, viscosity=viscosity)
    model.solve_columns = columns
    initial = State(.2, np.array([.4, -.2]), .3)
    trial = make_scheme(name, gamma=2.).step(model, initial, .03)
    info = scheme_info(name)
    assert calls == [2]*info.n_stages
    previous = initial.r
    for stage in trial.stages:
        assert stage.correction_predictor == previous
        assert len(stage.corrections) == info.n_corrections == (2 if info.order == 3 else 1)
        assert stage.r == stage.corrections[-1]
        z = previous if info.n_corrections == 1 else stage.corrections[-2]
        assert stage.frozen_factor == pytest.approx(sum(z**j for j in range(info.order)))
        assert stage.candidates == stage.root_residuals == []
        assert stage.scalar_residual < 1e-14 and stage.residual < 1e-14
        previous = stage.r


def test_two_updates_are_kept_even_when_the_first_is_exact():
    result = solve_scalar_stage(0., .1, 1., 0., 0., 0., order=3)
    assert result.values == (0., 0.) and result.factor == 1.

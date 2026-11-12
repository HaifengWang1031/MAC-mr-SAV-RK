"""Cached FEM quadrature at public projection/nonlinear/diagnostic interfaces."""
import numpy as np
from scipy.linalg import solve

from solver.fem import FEMModel, Space


def test_projection_evaluates_callable_once_per_grid_without_assuming_time_modes():
    model = FEMModel(Space(4, 3), .1)
    calls = []
    amplitude = [1.]

    def field(x, y):
        calls.append(x.shape)
        return amplitude[0]*(x+2*y), amplitude[0]*(-x+y)

    coefficients = model.project(field)
    assert len(calls) == 1
    x, y = model.velocity_basis.doflocs
    full = np.zeros(model.velocity_basis.N)
    for values, indices in zip((x+2*y, -x+y), model.component_dofs):
        full[indices] = values[indices]
    expected = solve(model.mass.toarray(), (model.full_mass@full)[model.free_dofs], assume_a='pos')
    np.testing.assert_allclose(coefficients, expected, atol=2e-13)
    amplitude[0] = 2.
    np.testing.assert_allclose(model.project(field), 2*expected, atol=3e-13)
    assert len(calls) == 2


def test_field_cache_tracks_mutable_inputs_and_does_not_expose_mutable_loads():
    model = FEMModel(Space(4, 3), .1)
    velocity = model.project(lambda x, y: (x*(1-x)*y*(1-y), 0.))
    nonlinear = model.nonlinear(velocity)
    expected = nonlinear.copy()
    assert model.max_abs(expected) > 1e-5
    nonlinear[0] += 1.
    np.testing.assert_allclose(model.nonlinear(velocity), expected, atol=1e-13)
    diagnostic = model.diagnostics(model.state(0., velocity))
    velocity *= 2.
    np.testing.assert_allclose(model.nonlinear(velocity), 4*expected, atol=2e-13)
    changed = model.diagnostics(model.state(0., velocity))
    np.testing.assert_allclose(changed['divergence_inf'], 2*diagnostic['divergence_inf'], atol=1e-13)
    np.testing.assert_allclose(changed['kinetic'], 4*diagnostic['kinetic'], atol=1e-14)
    assert model.performance_metrics()['volume_field_cache_hits'] > 0


def test_divergence_only_kernel_preserves_the_full_quadrature_diagnostic():
    model = FEMModel(Space(4, 3), .1)
    basis = model.velocity_basis
    full = np.zeros(basis.N)
    x, y = basis.doflocs
    for values, indices in zip((x*x+x*y, y*y+x*y), model.component_dofs):
        full[indices] = values[indices]
    before = full.copy()
    q = model._quadrature
    actual = q.divergence(full)
    qx, qy = basis.global_coordinates()
    np.testing.assert_allclose(actual, 3*(qx+qy), rtol=0., atol=2e-13)
    assert q.metrics()['field_evaluations'] == 0
    _, gradient = q.evaluate(full)
    np.testing.assert_allclose(q.divergence(full), gradient[0, 0]+gradient[1, 1], atol=2e-13)
    np.testing.assert_array_equal(full, before)


def test_single_response_reuses_its_field_for_stage_diagnostics_and_convection():
    model = FEMModel(Space(4, 3), .1)
    rhs = model.project(lambda x, y: (y*(1-y), x*(2-x)))
    result = model.solve(rhs, mass=1., viscosity=.003)
    assert np.linalg.norm(result.velocity) > .01
    before = model.performance_metrics()['volume_field_evaluations']
    model.stage_divergences(result.velocity)
    model.nonlinear(result.velocity)
    assert model.performance_metrics()['volume_field_evaluations'] == before

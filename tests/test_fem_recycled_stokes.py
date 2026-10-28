"""Exact current shifted equations, full pressure rows, and bounded recycled factors."""
from typing import Any

import numpy as np
import pytest
from scipy.linalg import null_space

from solver.fem import FEMModel, Space
import solver.fem.stokes as stokes_module


@pytest.mark.parametrize('backend', ['direct', 'recycled'])
def test_shifted_multicolumn_solves_have_independently_constructed_solutions(backend: str):
    model = FEMModel(Space(4, 3), .1, cache_size=2, stokes_backend=backend)
    # Independent dense SVD defines two exact discrete solenoidal velocities.
    null = null_space(model.divergence.toarray())
    velocities = [null[:, 0], null[:, 1]]
    x, y = model.pressure_basis.doflocs
    pressures = [x+2*y-1.5, -3*x+y+1.]
    shifts = [(1., .003), (1., .00303), (1., .003045), (1., .00301),
              (1., .003), (0., .1), (1., 0.)]
    for alpha, beta in shifts:
        rhs = [model.combine((alpha, v), (beta, model.apply_K(v)), (1., model.apply_G(p)))
               for v, p in zip(velocities, pressures)]
        copies = [v.copy() for v in rhs]
        for value in rhs:
            value.flags.writeable = False
        results = model.solve_columns(rhs, mass=alpha, viscosity=beta)
        for k, result in enumerate(results):
            np.testing.assert_allclose(result.velocity, velocities[k], rtol=0., atol=3e-9)
            np.testing.assert_allclose(result.pressure, pressures[k], rtol=0., atol=3e-9)
            assert model.max_abs(model.apply_D(result.velocity)) < 1e-9
            assert abs(model.pressure_integral(result.pressure)) < 1e-9
            np.testing.assert_array_equal(rhs[k], copies[k])
    metrics = model.stokes.metrics()
    assert metrics['linear_columns'] == 2*len(shifts) and metrics['cached_factors'] <= 2
    assert metrics['max_coefficient_residual'] < 1e-9
    if backend == 'recycled':
        assert metrics['recycled_solves'] > 0 and metrics['factorizations'] < len(shifts)


def test_fast_factor_residual_failure_retries_robust_factor(monkeypatch: pytest.MonkeyPatch):
    real_splu = stokes_module.splu

    class InaccurateExternalFactor:
        def __init__(self, factor: Any) -> None:
            self.factor = factor

        def solve(self, rhs: np.ndarray) -> np.ndarray:
            result = self.factor.solve(rhs)
            result[0] += 1.
            return result

    def external_factor(matrix: Any, **options: Any) -> Any:
        factor = real_splu(matrix, **options)
        return factor if options.get('permc_spec') == 'COLAMD' else InaccurateExternalFactor(factor)

    monkeypatch.setattr(stokes_module, 'splu', external_factor)
    model = FEMModel(Space(3, 3), .1)
    for _ in range(2):
        result = model.solve(model.zero_velocity(), mass=1., viscosity=.001)
        np.testing.assert_array_equal(result.velocity, 0.)
        np.testing.assert_array_equal(result.pressure, 0.)
    metrics = model.stokes.metrics()
    assert metrics['fallbacks'] == 1 and metrics['factorizations'] == 2

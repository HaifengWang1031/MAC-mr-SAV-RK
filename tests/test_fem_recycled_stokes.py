"""Exact current shifted equations, full pressure rows, and bounded recycled factors."""
from typing import Any
import ctypes

import numpy as np
import pytest
from scipy.linalg import null_space

from solver.fem import FEMModel, Space
import solver.fem.stokes as stokes_module


def test_native_local_thread_limit_restores_callers_setting_on_success_and_error():
    pytest.importorskip('pypardiso')
    from pypardiso import PyPardisoSolver
    from scipy.sparse import csc_matrix
    from solver.fem.pardiso import PardisoFactor
    library = PyPardisoSolver().libmkl
    set_local = library.MKL_Set_Num_Threads_Local
    set_local.argtypes, set_local.restype = [ctypes.c_int], ctypes.c_int
    get_threads = library.MKL_Get_Max_Threads
    get_threads.argtypes, get_threads.restype = [], ctypes.c_int
    previous = set_local(3)
    try:
        factor = PardisoFactor(csc_matrix([[2., 1.], [1., 3.]]), threads=2)
        assert get_threads() == 3
        np.testing.assert_allclose(factor.solve(np.array([4., 7.])), [1., 2.], atol=2e-15)
        assert get_threads() == 3
        with pytest.raises(RuntimeError):
            factor.solve(np.ones(3))
        assert get_threads() == 3
    finally:
        set_local(previous)


@pytest.mark.parametrize('backend', ['direct', 'recycled'])
@pytest.mark.parametrize(('factor_backend', 'ordering'), [('superlu', 'mmd'), ('superlu', 'metis'), ('pardiso', 'auto')])
def test_shifted_multicolumn_solves_have_independently_constructed_solutions(backend: str, factor_backend: str, ordering: str):
    if factor_backend == 'pardiso':
        pytest.importorskip('pypardiso')
    model = FEMModel(Space(4, 3), .1, cache_size=2, stokes_backend=backend, stokes_ordering=ordering,
                     stokes_factor_backend=factor_backend, stokes_threads=2 if factor_backend == 'pardiso' else None)
    # Independent dense SVD defines two exact discrete solenoidal velocities.
    null = null_space(model.divergence.toarray())
    velocities = [null[:, 0], null[:, 1], np.zeros(null.shape[0])]
    x, y = model.pressure_basis.doflocs
    pressures = [x+2*y-1.5, -3*x+y+1., np.zeros_like(x)]
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
    assert metrics['linear_columns'] == 3*len(shifts) and metrics['cached_factors'] <= 2
    assert metrics['max_coefficient_residual'] < 1e-9
    if backend == 'recycled':
        assert metrics['recycled_solves'] > 0 and metrics['factorizations'] < len(shifts)
    if ordering == 'metis':
        assert metrics['ordering_preparations'] == 1
        assert metrics['metis_factorizations'] > 0
    if factor_backend == 'pardiso':
        assert metrics['pardiso_factorizations'] > 0 and metrics['fallbacks'] == 0


def test_stagnating_batch_correction_falls_back_to_checked_gmres(monkeypatch: pytest.MonkeyPatch):
    model = FEMModel(Space(4, 3), .1, cache_size=2, stokes_backend='recycled')
    velocity = null_space(model.divergence.toarray())[:, 0]
    x, y = model.pressure_basis.doflocs
    pressure = x+2*y-1.5
    def rhs(shift: float) -> np.ndarray:
        return model.combine((1., velocity), (shift, model.apply_K(velocity)), (1., model.apply_G(pressure)))
    model.solve(rhs(.003), mass=1., viscosity=.003)
    original = model.stokes._solve_factor
    def no_batch_progress(factor: Any, values: np.ndarray, *, preconditioner: bool = False) -> np.ndarray:
        if preconditioner and values.ndim == 2:
            return np.zeros_like(values)
        return original(factor, values, preconditioner=preconditioner)
    monkeypatch.setattr(model.stokes, '_solve_factor', no_batch_progress)
    result = model.solve(rhs(.00303), mass=1., viscosity=.00303)
    np.testing.assert_allclose(result.velocity, velocity, rtol=0., atol=3e-9)
    np.testing.assert_allclose(result.pressure, pressure, rtol=0., atol=3e-9)
    metrics = model.stokes.metrics()
    assert metrics['defect_failures'] == 1 and metrics['gmres_solves'] == 1
    assert metrics['factorizations'] == 1 and metrics['max_continuity_residual'] < 1e-9


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


def test_ordering_failure_preserves_checked_original_system(monkeypatch: pytest.MonkeyPatch):
    def unavailable(*args: Any) -> Any:
        raise RuntimeError('ordering unavailable')
    monkeypatch.setattr(stokes_module, 'nested_dissection', unavailable)
    model = FEMModel(Space(3, 3), .1, stokes_ordering='metis')
    x, y = model.pressure_basis.doflocs
    pressure = x+2*y-1.5
    for shift in (.001, .002):
        result = model.solve(model.apply_G(pressure), mass=1., viscosity=shift)
        np.testing.assert_allclose(result.velocity, 0., atol=3e-12)
        np.testing.assert_allclose(result.pressure, pressure, atol=3e-11)
    metrics = model.stokes.metrics()
    assert metrics['ordering_failures'] == metrics['ordering_preparations'] == 1
    assert metrics['metis_factorizations'] == 0 and metrics['factorizations'] == 2


def test_structural_ordering_is_shared_across_models_and_zero_shifts(monkeypatch: pytest.MonkeyPatch):
    import pymetis
    import solver.fem.ordering as ordering_module
    ordering_module._CACHE.clear()
    original, calls = pymetis.nested_dissection, []
    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(pymetis, 'nested_dissection', counted)
    first = FEMModel(Space(5, 3), .1, stokes_ordering='metis')
    second = FEMModel(Space(5, 3), .1, stokes_ordering='metis')
    for model, key in ((first, (0., .1)), (first, (1., 0.)), (second, (1., .003))):
        x, y = model.pressure_basis.doflocs
        pressure = x+2*y-1.5
        result = model.solve(model.apply_G(pressure), mass=key[0], viscosity=key[1])
        np.testing.assert_allclose(result.velocity, 0., atol=3e-12)
        np.testing.assert_allclose(result.pressure, pressure, atol=3e-11)
    assert len(calls) == 1
    assert first.stokes.metrics()['ordering_preparations'] == 1
    assert second.stokes.metrics()['ordering_cache_hits'] == 1

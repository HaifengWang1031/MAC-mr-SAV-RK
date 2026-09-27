import numpy as np
from solver.mac.grid import MACGrid
from solver.mac.operators import MACOperators
from solver.mac.stokes import DirectStokes
from solver.mac.tensor_stokes import TensorStokes

def test_pressure_gradient_produces_zero_velocity_on_rectangular_grid():
    grid = MACGrid(7, 5, lx=2.0, ly=1.0)
    ops = MACOperators(grid)
    p = np.arange(grid.np, dtype=float)**2
    p -= p.mean()
    backend = DirectStokes(ops, cache_size=2)
    result = backend.solve(ops.G @ p, mass=1.0, viscosity=0.07)
    assert np.max(np.abs(result.velocity)) < 1e-10
    np.testing.assert_allclose(result.pressure.ravel(), p, atol=1e-9)
    assert result.divergence_inf < 1e-10
    assert result.residual < 1e-10
    assert abs(result.pressure.mean()) < 1e-12

def test_multi_rhs_and_bounded_factorization_cache():
    grid=MACGrid(7,5)
    ops=MACOperators(grid)
    solver=DirectStokes(ops,cache_size=2)
    rhs=np.random.default_rng(12).normal(size=(grid.size,2))
    both=solver.solve(rhs,mass=1.,viscosity=.1)
    for i in range(2):
        single=solver.solve(rhs[:,i],mass=1.,viscosity=.1)
        np.testing.assert_allclose(single.velocity,both.velocity[:,i],atol=1e-12)
    assert solver.factorizations==1
    solver.solve(rhs,mass=1.,viscosity=.2)
    solver.solve(rhs,mass=1.,viscosity=.3)
    assert len(solver.cache)==2
    solver.solve(rhs,mass=1.,viscosity=.1)
    assert solver.factorizations==4


def test_tensor_stokes_matches_direct_for_rectangular_multi_rhs_and_limit_shifts():
    grid = MACGrid(7, 5, lx=2.0, ly=1.0)
    ops = MACOperators(grid)
    rhs = np.random.default_rng(19).normal(size=(grid.size, 2))
    tensor = TensorStokes(ops, tolerance=1e-10)
    for mass, viscosity in ((1.0, 0.07), (1.0, 0.0), (0.0, 0.07)):
        expected = DirectStokes(ops, tolerance=1e-10).solve(
            rhs, mass=mass, viscosity=viscosity
        )
        actual = tensor.solve(rhs, mass=mass, viscosity=viscosity)
        np.testing.assert_allclose(actual.velocity, expected.velocity, atol=3e-11)
        np.testing.assert_allclose(actual.pressure, expected.pressure, atol=3e-11)
        assert actual.residual < 1e-10
        assert actual.divergence_inf < 1e-10
        assert abs(actual.pressure.mean(axis=(0, 1))).max() < 1e-12


def test_tensor_stokes_reuses_setup_across_shifted_gradient_solves():
    grid = MACGrid(8, 6)
    ops = MACOperators(grid)
    rng = np.random.default_rng(23)
    pressure = rng.normal(size=grid.np)
    pressure -= pressure.mean()
    solver = TensorStokes(ops, tolerance=1e-10)
    setup_seconds = solver.setup_seconds
    for viscosity in (0.02, 0.05, 0.11):
        result = solver.solve(ops.G @ pressure, mass=1.0, viscosity=viscosity)
        np.testing.assert_allclose(result.velocity, 0.0, atol=2e-11)
        np.testing.assert_allclose(result.pressure.ravel(), pressure, atol=2e-10)
    assert solver.setup_seconds == setup_seconds
    assert solver.solves == 3
    assert solver.total_iterations > 0


def test_matrix_free_tensor_stokes_matches_sparse_operator_path():
    grid = MACGrid(11, 7, lx=2.0, ly=1.0)
    ops = MACOperators(grid)
    rhs = np.random.default_rng(37).normal(size=(grid.size, 2))
    matrix_free = TensorStokes(
        ops, tolerance=1e-10, matrix_free_operators=True
    )
    sparse = TensorStokes(ops, tolerance=1e-10, matrix_free_operators=False)

    for mass, viscosity in ((1.0, 0.03), (0.7, 0.09), (0.0, 0.05)):
        actual = matrix_free.solve(rhs, mass=mass, viscosity=viscosity)
        expected = sparse.solve(rhs, mass=mass, viscosity=viscosity)
        np.testing.assert_allclose(actual.velocity, expected.velocity, atol=2e-11)
        np.testing.assert_allclose(actual.pressure, expected.pressure, atol=2e-11)
        assert matrix_free.last_iterations == sparse.last_iterations
        assert actual.residual < 1e-10
        assert actual.divergence_inf < 1e-10


def test_reused_transform_buffers_preserve_rhs_and_solution():
    grid = MACGrid(11, 7, lx=2.0, ly=1.0)
    ops = MACOperators(grid)
    rhs = np.random.default_rng(43).normal(size=(grid.size, 2))
    original = rhs.copy()
    reused = TensorStokes(
        ops,
        tolerance=1e-10,
        matrix_free_operators=True,
        reuse_transform_buffers=True,
    )
    baseline = TensorStokes(
        ops,
        tolerance=1e-10,
        matrix_free_operators=True,
        reuse_transform_buffers=False,
    )

    actual = reused.solve(rhs, mass=1.0, viscosity=0.05)
    expected = baseline.solve(rhs, mass=1.0, viscosity=0.05)

    np.testing.assert_array_equal(rhs, original)
    np.testing.assert_allclose(actual.velocity, expected.velocity, atol=2e-11)
    np.testing.assert_allclose(actual.pressure, expected.pressure, atol=2e-11)
    assert reused.last_iterations == baseline.last_iterations

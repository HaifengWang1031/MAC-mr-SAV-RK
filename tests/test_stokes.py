import numpy as np
from solver.mac.grid import MACGrid
from solver.mac.operators import MACOperators
from solver.mac.stokes import DirectStokes

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

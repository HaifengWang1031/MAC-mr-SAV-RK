"""Independent divergence and semidiscrete residual for the manufactured MAC field."""
import numpy as np
from experiments.manufactured_convergence.model import amplitude, amplitude_derivative, configure, spatial_velocity
from solver.mac.grid import MACGrid
from solver.mac_ns import MACNavierStokes


def test_discrete_manufactured_velocity_satisfies_mac_equation():
    grid=MACGrid(12,10)
    model=MACNavierStokes(grid,.01)
    shape=spatial_velocity(grid)
    assert np.max(np.abs(model.apply_D(shape)))<1e-12
    initial=configure(model,.05)
    np.testing.assert_allclose(initial,.05*shape,rtol=0,atol=1e-15)
    for time in (0.,.37,2.):
        a=amplitude(time,.05)
        velocity=a*shape
        residual=amplitude_derivative(time,.05)*shape+model.nu*model.apply_K(velocity) \
                 +model.nonlinear(velocity)-model.force(time)
        assert grid.norm(residual)<1e-12

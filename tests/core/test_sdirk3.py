"""Independent ODE order, steady MAC and spectral four-stage checks."""
import numpy as np

from solver.integrate import integrate
from solver.schemes.sdirk3 import C, D, DHAT, SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV
from tests.references.ode import ODEModel, State


def test_coefficients_and_third_order_on_independent_ode():
    assert np.allclose(D.sum(axis=1),DHAT.sum(axis=1))
    np.testing.assert_allclose(D.sum(axis=1),np.diff(C),atol=1e-15)
    model=ODEModel()
    for scheme in (SDIRK3(),SDIRK3MRSAV()):
        errors=[]
        for count in (50,100,200):
            result=integrate(model,scheme,State(0.,model.exact(0.)),[2/count]*count)
            assert result.status=='complete',result.error
            assert len(result.final_stages)==4
            errors.append(np.linalg.norm(result.final.velocity-model.exact(2.)))
        assert min(np.log2(np.array(errors[:-1])/errors[1:]))>2.85,(scheme.name,errors)


def test_steady_mac_velocity_and_zero_scalar_are_preserved():
    from solver.mac.grid import MACGrid
    from solver.mac_ns import MACNavierStokes
    grid = MACGrid(8, 6)
    model = MACNavierStokes(grid, .1)
    drive = np.random.default_rng(7).normal(size=grid.size)
    steady = model.backend.solve(drive,mass=1.,viscosity=.1).velocity
    force = model.nu*model.apply_K(steady)+model.nonlinear(steady)
    model.force = lambda t: force
    state = model.state(0.,steady)
    for method in (SDIRK3(),SDIRK3MRSAV()):
        trial=method.step(model,state,.013)
        assert grid.norm(model.vector(trial.state)-steady)<1e-11
        assert abs(trial.state.r)<1e-11


def test_spectral_lifting_runs_all_four_stages():
    from solver.spectral.assembly import Space
    from solver.spectral.lifting import LidLifting
    from solver.spectral.model import SpectralModel
    space=Space(4,1.,1.)
    model=SpectralModel(space,.1,lifting=LidLifting(space))
    initial=model.state(0.,model.zero_velocity())
    for method in (SDIRK3(),SDIRK3MRSAV()):
        trial=method.step(model,initial,.01)
        assert len(trial.stages)==4
        assert np.isfinite(model.vector(trial.state)).all()
        assert max(stage.continuity_residual for stage in trial.stages)<1e-10

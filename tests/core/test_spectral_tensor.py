"""Known coupled solutions and tensor/Schur validation across shifts and liftings."""
import numpy as np
import pytest

from solver.integrate import integrate
from solver.schemes.registry import SCHEMES, make_scheme
from solver.schemes.registry import scheme_info
from solver.adaptivity import IController, PIController, integrate_adaptive
from solver.spectral.assembly import Space
from solver.spectral.lifting import LidLifting
from solver.spectral.model import SpectralModel
from solver.spectral.stokes import SpectralStokes
from solver.spectral.tensor_stokes import TensorStokes


@pytest.mark.parametrize('mass,viscosity', [(1., 0.), (1., 1e-6), (1., .2), (0., .1)])
def test_tensor_stokes_recovers_known_velocity_pressure_and_constraint(mass, viscosity):
    solver = TensorStokes(Space(8, 1.3, .8), tolerance=1e-11, krylov_tolerance=1e-13)
    random = np.random.default_rng(59)
    u, v, p = random.normal(size=(3, 64))
    p[0] = 0.
    h = mass*solver.mass+viscosity*solver.stiffness
    fx, fy = h @ u-solver.divergence_x.T @ p, h @ v-solver.divergence_y.T @ p
    constraint = solver.divergence_x @ u+solver.divergence_y @ v
    solution = solver.solve(fx, fy, mass=mass, viscosity=viscosity, continuity=constraint)
    np.testing.assert_allclose(solution.velocity_x.ravel(), u, atol=2e-11)
    np.testing.assert_allclose(solution.velocity_y.ravel(), v, atol=2e-11)
    np.testing.assert_allclose(solution.pressure.ravel(), p, atol=2e-11)
    assert solver.factorizations == 0 and solver.total_iterations > 0


@pytest.mark.parametrize('scheme', SCHEMES)
def test_tensor_and_direct_integrations_agree_with_lid_and_variable_steps(scheme):
    space = Space(8, 1.3, .8)
    models = [SpectralModel(space, .1, lifting=LidLifting(space), stokes_backend=backend)
              for backend in ('direct', 'tensor')]
    results = [integrate(model, make_scheme(scheme, 1.), model.state(0., model.zero_velocity()),
                         [.001, .0015, .0007]) for model in models]
    assert all(result.status == 'complete' for result in results)
    np.testing.assert_allclose(models[0].vector(results[0].final), models[1].vector(results[1].final),
                               rtol=1e-10, atol=1e-12)
    assert results[0].final.r == pytest.approx(results[1].final.r, abs=1e-12)
    for first, second in zip(results[0].final_stages, results[1].final_stages):
        np.testing.assert_allclose(first.pressure, second.pressure, rtol=1e-9, atol=1e-11)
    assert max(s['continuity_residual'] for stages in results[1].stages for s in stages) < 1e-10


def test_tensor_schur_exposes_iteration_failure_instead_of_accepting_unchecked_solution():
    solver = TensorStokes(Space(8), max_iterations=1, krylov_tolerance=1e-14)
    with pytest.raises(RuntimeError, match='Schur CG failed'):
        solver.solve(np.random.default_rng(4).normal(size=64), np.zeros(64), mass=1., viscosity=.01)


@pytest.mark.parametrize('scheme', ['sdirk2', 'sdirk2_mrsav', 'sdirk3', 'sdirk3_mrsav'])
@pytest.mark.parametrize('controller_class', [IController, PIController])
def test_tensor_matches_direct_adaptive_rejections_and_endpoint(scheme, controller_class):
    space = Space(8, 1.3, .8)
    models, results = [], []
    for backend in ('direct', 'tensor'):
        model = SpectralModel(space, .1, stokes_backend=backend)
        def u(x, y):
            a, b = x/space.lx, y/space.ly
            return 20*a**2*(1-a)**2*b*(1-b)*(1-2*b)/space.ly
        def v(x, y):
            a, b = x/space.lx, y/space.ly
            return -20*a*(1-a)*(1-2*a)*b**2*(1-b)**2/space.lx
        initial = model.state(0., np.r_[model.project(u), model.project(v)])
        order = scheme_info(scheme).estimator_order
        assert order is not None
        controller = controller_class(atol=1e-10, rtol=1e-7, min_step=1e-10, estimator_order=order)
        result = integrate_adaptive(model, make_scheme(scheme, 1000.), initial, .02, .02,
                                    controller=controller)
        models.append(model)
        results.append(result)
    assert all(result.status == 'complete' and result.final.t == .02 for result in results)
    assert any(not attempt['accepted'] for attempt in results[0].attempts)
    np.testing.assert_array_equal([a['accepted'] for a in results[0].attempts],
                                  [a['accepted'] for a in results[1].attempts])
    # The final forced remainder accumulates previous step differences; compare it absolutely.
    np.testing.assert_allclose([a['step'] for a in results[0].attempts[:-1]],
                               [a['step'] for a in results[1].attempts[:-1]], rtol=2e-6, atol=1e-12)
    assert results[0].attempts[-1]['step'] == pytest.approx(results[1].attempts[-1]['step'], abs=1e-9)
    np.testing.assert_allclose(models[0].vector(results[0].final), models[1].vector(results[1].final),
                               rtol=1e-8, atol=1e-11)
    assert results[0].final.r == pytest.approx(results[1].final.r, abs=1e-11)

"""FEM behavior at the public Model/step seam; references are analytic fields."""
import numpy as np
import sympy as sy
import pytest
import scipy.sparse as sp

from solver.fem import FEMModel, Space
from solver.core import State
from solver.integrate import integrate
from solver.schemes.sdirk2 import SDIRK2
from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV
from solver.fem.mass import VelocityMassFactor


@pytest.mark.parametrize('coupled', [False, True])
def test_consistent_mass_component_factor_restores_all_coefficients(coupled: bool):
    # An independently specified SPD form in interleaved vector coordinates.
    matrix = np.array([[2., 0., 1., 0.], [0., 2., 0., 1.],
                       [1., 0., 3., 0.], [0., 1., 0., 3.]])
    if coupled:
        matrix[0, 1] = matrix[1, 0] = .2
    factor = VelocityMassFactor(sp.csc_matrix(matrix), [np.array([0, 2]), np.array([1, 3])])
    known = np.array([[1., 0., -.3], [2., 1., .5], [3., -1., .7], [4., 2., -.9]])
    for values in (known[:, 0], known):
        rhs = matrix @ values
        before = rhs.copy()
        rhs.flags.writeable = False
        np.testing.assert_allclose(factor.solve(rhs), values, atol=2e-15)
        np.testing.assert_array_equal(rhs, before)
    assert len(factor.components) == (0 if coupled else 2)


def test_steady_stokes_linear_pressure_on_rectangle():
    model = FEMModel(Space(4, 3, lx=2., ly=1.), nu=.2)
    force = model.project(lambda x, y: (np.ones_like(x), 2*np.ones_like(y)))
    solution = model.solve(force, mass=0., viscosity=model.nu)
    points = np.array([[.13, .91, 1.8], [.2, .73, .4]])
    np.testing.assert_allclose(model.evaluate_velocity(solution.velocity, points), 0., atol=2e-13)
    np.testing.assert_allclose(model.evaluate_pressure(solution.pressure, points),
                               points[0]+2*points[1]-2., atol=2e-12)
    assert abs(model.pressure_integral(solution.pressure)) < 1e-13
    assert solution.residual < 1e-12
    assert model.max_abs(model.apply_D(solution.velocity)) < 1e-13


def test_p2_edge_field_mass_gradient_pairings_and_state_roundtrip():
    model = FEMModel(Space(2, 2, lx=2., ly=1.), nu=.2)
    coordinates = model.velocity_basis.doflocs[:, model.component_dofs[0]]
    u = np.zeros(coordinates.shape[1])
    u[np.all(coordinates == np.array([[.5], [.5]]), axis=0)] = 1.
    state = State(.3, u, np.zeros_like(u), .17)
    velocity = model.vector(state)
    reconstructed = model.state(state.t, velocity, state.r)
    np.testing.assert_array_equal(reconstructed.u, u)
    np.testing.assert_array_equal(reconstructed.v, state.v)
    assert reconstructed.t == .3 and reconstructed.r == .17
    # One edge basis 4*lambda_i*lambda_j on two triangles of area 1/4.
    np.testing.assert_allclose(model.inner(velocity, velocity), 4/45, atol=2e-15)
    np.testing.assert_allclose(model.inner(velocity, model.apply_K(velocity)), 20/3, atol=2e-13)
    point = np.array([[.25], [.4]])
    np.testing.assert_allclose(model.evaluate_velocity(velocity, point)[:, 0], [.55, 0.], atol=2e-15)
    np.testing.assert_allclose(model.evaluate_gradient(velocity, point)[:, :, 0], [[1.2, 2.], [0., 0.]], atol=2e-14)
    wall_points = np.array([[0., .3, 2., .7], [.4, 0., .3, 1.]])
    np.testing.assert_allclose(model.evaluate_velocity(velocity, wall_points), 0., atol=2e-15)
    other = np.random.default_rng(5).normal(size=velocity.size)
    pressure = np.random.default_rng(6).normal(size=model.zero_pressure().size)
    np.testing.assert_allclose(model.inner(velocity, model.apply_K(other)),
                               model.inner(model.apply_K(velocity), other), atol=2e-13)
    np.testing.assert_allclose(model.inner(other, model.apply_G(pressure)),
                               -pressure @ model.apply_D(other), atol=2e-13)
    diagnostics = model.diagnostics(state)
    np.testing.assert_allclose(diagnostics['kinetic'], 2/45, atol=2e-15)
    np.testing.assert_allclose(diagnostics['h1_seminorm_squared'], 20/3, atol=2e-13)


def test_skew_convection_has_known_edge_vertex_pairing_and_exact_quadrature():
    model = FEMModel(Space(2, 2, lx=2., ly=1., quadrature_order=5), nu=.2)
    coordinates = model.velocity_basis.doflocs[:, model.component_dofs[0]]
    edge = np.all(coordinates == np.array([[.5], [.5]]), axis=0).astype(float)
    vertex = np.all(coordinates == np.array([[1.], [.5]]), axis=0).astype(float)
    velocity = model.vector(State(0., edge, np.zeros_like(edge)))
    test = model.vector(State(0., vertex, np.zeros_like(vertex)))
    convection, lifting = model.nonlinear_with_lifting(velocity)
    # Direct barycentric integration of b(4 li lj, 4 li lj, lj(2lj-1)).
    np.testing.assert_allclose(model.inner(convection, test), -1/21, atol=2e-14)
    np.testing.assert_allclose(model.inner(convection, velocity), 0., atol=2e-14)
    assert lifting == 0.
    assert model.physical_divergence_inf(velocity) > 1.
    high = FEMModel(Space(2, 2, lx=2., ly=1., quadrature_order=9), nu=.2)
    np.testing.assert_allclose(convection, high.nonlinear(velocity), atol=2e-13)
    random_velocity = np.random.default_rng(19).normal(size=velocity.size)
    assert abs(model.inner(model.nonlinear(random_velocity), random_velocity)) < 2e-13


def manufactured_stokes():
    x, y = sy.symbols('x y')
    psi = x**2*(2-x)**2*y**2*(1-y)**2
    velocity = (sy.diff(psi, y), -sy.diff(psi, x))
    pressure = x**2+2*x*y+sy.Rational(3, 10)*y**2-sy.Rational(73, 30)
    force = tuple(-sy.Rational(1, 5)*(sy.diff(u, x, 2)+sy.diff(u, y, 2))+sy.diff(pressure, z)
                  for u, z in zip(velocity, (x, y)))
    gradient = [[sy.diff(u, z) for z in (x, y)] for u in velocity]
    return tuple(sy.lambdify((x, y), field, 'numpy') for field in (velocity, gradient, pressure, force))


def test_manufactured_stokes_converges_and_distinguishes_weak_from_physical_divergence():
    exact_velocity, exact_gradient, exact_pressure, force = manufactured_stokes()
    errors = []
    for n in (2, 4, 8):
        model = FEMModel(Space(2*n, n, lx=2., ly=1., quadrature_order=8), nu=.2)
        solved = model.solve(model.project(force), mass=0., viscosity=.2)
        points = model.velocity_basis.global_coordinates().reshape(2, -1)
        weight = model.velocity_basis.dx.reshape(-1)
        x, y = points
        velocity_error = model.evaluate_velocity(solved.velocity, points)-exact_velocity(x, y)
        gradient_error = model.evaluate_gradient(solved.velocity, points)-exact_gradient(x, y)
        pressure_error = model.evaluate_pressure(solved.pressure, points)-exact_pressure(x, y)
        errors.append([np.sqrt(np.sum(velocity_error**2*weight)),
                       np.sqrt(np.sum(gradient_error**2*weight)),
                       np.sqrt(np.sum(pressure_error**2*weight)),
                       model.diagnostics(model.state(0., solved.velocity))['divergence_l2']])
        assert model.max_abs(model.apply_D(solved.velocity)) < 2e-12
        assert abs(model.pressure_integral(solved.pressure)) < 2e-12
        # P1 basis integrals differ at corners, edges and interior vertices.
        assert abs(np.mean(solved.pressure)) > .02
        assert solved.divergence_inf > 1e-4
    rates = np.log2(np.array(errors[:-1])/np.array(errors[1:]))
    assert np.all(rates[-1] > [2.7, 1.8, 1.8, 1.8]), (errors, rates)


def test_shared_matrix_columns_and_pressure_projection():
    model = FEMModel(Space(5, 3, lx=2., ly=1.), nu=.2, cache_size=1)
    first = model.project(lambda x, y: (x*(2-x)*y*(1-y), x*y))
    second = model.project(lambda x, y: (np.cos(x+3*y), np.sin(2*x-y)))
    solutions = model.solve_columns([first, second], mass=1.3, viscosity=.07)
    for rhs, solved in zip((first, second), solutions):
        single = model.solve(rhs, mass=1.3, viscosity=.07)
        np.testing.assert_allclose(solved.velocity, single.velocity, atol=2e-13)
        np.testing.assert_allclose(solved.pressure, single.pressure, atol=2e-13)
        defect = model.combine((1.3, solved.velocity), (.07, model.apply_K(solved.velocity)),
                               (1., model.apply_G(solved.pressure)), (-1., rhs))
        assert model.max_abs(defect) < 2e-11
        assert model.max_abs(model.apply_D(solved.velocity)) < 1e-12
        assert model.diagnostics(model.state(0., solved.velocity))['continuity_residual'] < 1e-12
        assert abs(model.pressure_integral(solved.pressure)) < 1e-12
    # Exercise another shift and return to the first (bounded cache must not change results).
    model.solve(first, mass=0., viscosity=.2)
    np.testing.assert_allclose(model.solve(first, mass=1.3, viscosity=.07).velocity,
                               solutions[0].velocity, atol=2e-13)
    pressure = model.project_pressure(lambda x, y: x+2*y)
    np.testing.assert_allclose(model.evaluate_pressure(pressure, np.array([[.2], [.7]])), [-.4], atol=2e-13)
    assert abs(model.pressure_integral(pressure)) < 1e-13


@pytest.mark.parametrize('scheme', [SDIRK2(), SDIRK2MRSAV(gamma=2.)], ids=['ordinary', 'SAV'])
def test_sdirk2_steps_preserve_input_and_decay_unforced_energy(scheme):
    model = FEMModel(Space(5, 3, lx=2., ly=1.), nu=.2)
    projected = model.project(lambda x, y: (np.sin(2*x)*np.sin(3*y), np.sin(x)*np.sin(2*y)))
    velocity = model.solve(projected, mass=1., viscosity=0.).velocity
    initial = model.state(.3, velocity)
    before_u, before_v = initial.u.copy(), initial.v.copy()
    trial = scheme.step(model, initial, .002)
    np.testing.assert_array_equal(initial.u, before_u)
    np.testing.assert_array_equal(initial.v, before_v)
    assert initial.t == .3 and initial.r == 0.
    assert trial.state.t == .302
    assert 0. < model.diagnostics(trial.state)['kinetic'] < model.diagnostics(initial)['kinetic']
    for stage in trial.stages:
        assert stage.residual < 1e-10
        assert stage.continuity_residual < 1e-12
        assert abs(model.pressure_integral(stage.pressure)) < 1e-10
        assert stage.scalar_residual < 1e-12
        if scheme.name.endswith('mrsav'):
            assert stage.candidates == stage.root_residuals == []
            assert stage.r == stage.corrections[-1]
            assert len(stage.corrections) == (2 if scheme.name == 'sdirk3_mrsav' else 1)


def test_nonpolynomial_force_quadrature_sensitivity():
    ordinary = FEMModel(Space(6, 4, lx=2., ly=1., quadrature_order=6), nu=.2)
    higher = FEMModel(Space(6, 4, lx=2., ly=1., quadrature_order=10), nu=.2)
    function = lambda x, y: (np.sin(x+2*y), np.cos(2*x-y))
    first, second = ordinary.project(function), higher.project(function)
    reference = FEMModel(Space(6, 4, lx=2., ly=1., quadrature_order=12), nu=.2).project(function)
    low_error = np.sqrt(ordinary.inner(first-reference, first-reference))
    high_error = np.sqrt(ordinary.inner(second-reference, second-reference))
    assert high_error < 1e-10 and high_error < low_error/100


def test_constant_field_projection_and_invalid_boundaries():
    model = FEMModel(Space(3, 2), nu=.2)
    constant = model.project(lambda x, y: (1., 0.))
    assert np.isfinite(constant).all() and model.inner(constant, constant) > 0.
    np.testing.assert_allclose(model.project_pressure(lambda x, y: 3.), 0., atol=2e-14)
    for kwargs in ({'nx': 0, 'ny': 2}, {'nx': 2.5, 'ny': 2}, {'nx': True, 'ny': 2},
                   {'nx': 2, 'ny': 2, 'lx': np.nan}, {'nx': 2, 'ny': 2, 'ly': -1.},
                   {'nx': 2, 'ny': 2, 'quadrature_order': 4}):
        with pytest.raises(ValueError):
            Space(**kwargs)
    for nu, tolerance, cache_size in ((0., 1e-9, 4), (np.inf, 1e-9, 4),
                                      (.2, np.nan, 4), (.2, 1e-9, 0)):
        with pytest.raises(ValueError):
            FEMModel(Space(2, 2), nu, tolerance=tolerance, cache_size=cache_size)
    with pytest.raises(ValueError, match='pressure constraints'):
        FEMModel(Space(1, 1), nu=.2)
    for mass, viscosity in ((0., 0.), (-1., .1), (1., -1.), (np.nan, .1)):
        with pytest.raises(ValueError):
            model.solve(constant, mass=mass, viscosity=viscosity)
    with pytest.raises(ValueError):
        model.solve(np.full_like(constant, np.nan), mass=1., viscosity=.1)
    with pytest.raises(ValueError):
        model.solve(np.zeros(constant.size+1), mass=1., viscosity=.1)
    with pytest.raises(ValueError):
        model.project(lambda x, y: (x*np.nan, y))
    with pytest.raises(ValueError):
        model.project_pressure(lambda x, y: x*np.inf)
    with pytest.raises(ValueError):
        model.state(np.nan, constant)
    state = model.state(0., constant)
    state.u[0] = 1.
    with pytest.raises(ValueError, match='boundary'):
        model.vector(state)
    with pytest.raises(ValueError):
        model.evaluate_gradient(constant, np.array([[np.nan], [0.]]))
    with pytest.raises(ValueError):
        model.evaluate_velocity(constant, np.array([[2.], [.3]]))
    with pytest.raises(ValueError):
        model.apply_G(np.ones(model.zero_pressure().size+1))
    with pytest.raises(ValueError):
        model.combine((np.nan, constant))
    for dt in (0., -.1, np.nan):
        with pytest.raises(ValueError):
            SDIRK2().step(model, model.state(0., constant), dt)
    with pytest.raises((FloatingPointError, RuntimeError)):
        with np.errstate(over='ignore', invalid='ignore'):
            model.nonlinear(np.full_like(constant, 1e300))
    strict = FEMModel(Space(3, 2), nu=.2, tolerance=1e-30)
    with pytest.raises(RuntimeError, match='residual'):
        strict.solve(constant, mass=1., viscosity=.1)


def test_integrate_rejects_nonfinite_force_without_accepting_failed_step():
    model = FEMModel(Space(3, 2), nu=.2)
    initial = model.state(0., model.zero_velocity())
    force = model.project(lambda x, y: (np.sin(x), np.cos(y)))
    model.force = lambda t: force if t < .002 else np.full_like(force, np.nan)
    result = integrate(model, SDIRK2MRSAV(), initial, [.002, .002])
    assert result.status == 'failed'
    assert result.times == [0., .002]
    assert result.final.t == .002 and len(result.stages) == 1
    np.testing.assert_array_equal(model.vector(initial), 0.)

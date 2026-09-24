"""Spectral discretisation: basis, assembly and the steady Stokes solve.

The references are deliberately independent of the implementation: sympy integrates the
1D forms symbolically, and the 2D operators are compared against Gauss-Legendre
quadrature of the same weak forms evaluated through the basis evaluation routines. A
wrong scale factor, a wrong Kronecker order or a swapped direction all fail these before
any solver is involved.
"""
import numpy as np
import pytest
import sympy as sy
from scipy.fft import dstn, idstn
from solver.spectral import basis
from solver.spectral.assembly import Space, divergence_blocks, velocity_mass, velocity_stiffness
from solver.spectral.lifting import LidLifting
from solver.spectral.model import SpectralModel
from solver.spectral.stokes import SpectralStokes
from solver.schemes.sdirk2 import SDIRK2
from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV


def test_one_dimensional_matrices_match_symbolic_integration():
    size = 4
    xi = sy.symbols('xi')
    functions = [sy.legendre(k, xi) for k in range(size + 2)]
    phi = [functions[k] - functions[k + 2] for k in range(size)]
    mass = basis.mass_reference(size)
    stiffness = basis.stiffness_reference(size)
    divergence = basis.divergence_reference(size)
    projection = basis.projection_reference(size)
    for k in range(size):
        for l in range(size):
            assert abs(float(sy.integrate(phi[k] * phi[l], (xi, -1, 1))) - mass[k, l]) < 1e-13
            assert abs(float(sy.integrate(sy.diff(phi[k], xi) * sy.diff(phi[l], xi), (xi, -1, 1)))
                       - stiffness[k, l]) < 1e-13
            assert abs(float(sy.integrate(functions[k] * phi[l], (xi, -1, 1))) - projection[k, l]) < 1e-13
        for m in range(size):
            assert abs(float(sy.integrate(functions[m] * sy.diff(phi[k], xi), (xi, -1, 1)))
                       - divergence[m, k]) < 1e-13


def test_dirichlet_basis_vanishes_on_the_walls():
    space = Space(6, 1.3, .8)
    for direction, length in (('x', 1.3), ('y', .8)):
        values = basis.dirichlet_values(space.size, np.array([-1., 1.]))
        assert np.max(np.abs(values)) < 1e-14
    # the 2D field built from those bases is zero on all four sides
    rng = np.random.default_rng(0)
    coefficients = rng.standard_normal((space.size, space.size))
    for points in ((0., 1.3), (0., .8)):
        x = np.array([0., 1.3]) if points[1] == 1.3 else np.linspace(0, 1.3, 7)
        y = np.array([0., .8]) if points[0] == 0. else np.linspace(0, .8, 7)
        assert np.max(np.abs(space.evaluate(coefficients, x, y))) < 1e-12


@pytest.mark.parametrize('lx,ly', [(1.3, .8)])
def test_two_dimensional_operators_match_quadrature_of_the_weak_forms(lx, ly):
    """Every 2D block against quadrature of the same weak form, basis by basis.

    Written per basis pair rather than with batched matrix products: an earlier batched
    version paired the tensor indices wrongly and reported a 40% error for a correct
    assembly, which cost more time than the loop costs to run.
    """
    space = Space(4, lx, ly)
    nodes_x, weights_x, nodes_y, weights_y = space.nodes(extra=8)
    weights = np.outer(weights_y, weights_x)
    phi = basis.dirichlet_values(space.size, 2 * nodes_x / lx - 1)
    psi = basis.dirichlet_values(space.size, 2 * nodes_y / ly - 1)
    dphi = basis.dirichlet_derivatives(space.size, 2 * nodes_x / lx - 1) * (2 / lx)
    dpsi = basis.dirichlet_derivatives(space.size, 2 * nodes_y / ly - 1) * (2 / ly)
    chi = basis.legendre_values(space.size, 2 * nodes_x / lx - 1)[:space.size]
    eta = basis.legendre_values(space.size, 2 * nodes_y / ly - 1)[:space.size]
    modes = [(k, l) for k in range(space.size) for l in range(space.size)]

    def value(first, second):
        return np.outer(second, first)

    reference = {'mass': np.zeros((len(modes), len(modes))),
                 'stiffness': np.zeros((len(modes), len(modes))),
                 'divergence_x': np.zeros((len(modes), len(modes))),
                 'divergence_y': np.zeros((len(modes), len(modes)))}
    for a, (k, l) in enumerate(modes):
        for b, (m, n) in enumerate(modes):
            reference['mass'][a, b] = np.sum(weights * value(phi[k], psi[l]) * value(phi[m], psi[n]))
            reference['stiffness'][a, b] = np.sum(weights * (
                value(dphi[k], psi[l]) * value(dphi[m], psi[n])
                + value(phi[k], dpsi[l]) * value(phi[m], dpsi[n])))
            reference['divergence_x'][a, b] = np.sum(weights * value(chi[k], eta[l]) * value(dphi[m], psi[n]))
            reference['divergence_y'][a, b] = np.sum(weights * value(chi[k], eta[l]) * value(phi[m], dpsi[n]))
    divergence_x, divergence_y = divergence_blocks(space)
    assembled = {'mass': velocity_mass(space).toarray(), 'stiffness': velocity_stiffness(space).toarray(),
                 'divergence_x': divergence_x.toarray(), 'divergence_y': divergence_y.toarray()}
    for key in reference:
        scale = max(np.max(np.abs(reference[key])), 1e-30)
        assert np.max(np.abs(reference[key] - assembled[key])) < 1e-12 * scale, key


@pytest.mark.parametrize('lx,ly', [(1.3, .8)])
def test_manufactured_steady_stokes_converges_spectrally(lx, ly):
    """Spectral convergence of both fields against the manufactured solution.

    The rectangular case is the one that matters: the square domain cannot see a
    transposition of a mode-indexed array, because the fields and the basis are then
    symmetric under x <-> y.
    """
    from experiments.problems import _expressions
    nu, amplitude = .1, .2
    expressions = _expressions(lx, ly, nu)
    load_x = lambda x, y: amplitude * expressions[3](x, y, 0.)
    load_y = lambda x, y: amplitude * expressions[5](x, y, 0.)
    exact_x = lambda x, y: amplitude * expressions[0](x, y, 0.)
    exact_y = lambda x, y: amplitude * expressions[1](x, y, 0.)
    exact_p = lambda x, y: amplitude * expressions[2](x, y, 0.)
    errors_u, errors_p = [], []
    for size in (8, 12, 16, 20):
        space = Space(size, lx, ly)
        solver = SpectralStokes(space, tolerance=1e-11)
        solution = solver.solve(space.load(load_x), space.load(load_y), mass=0., viscosity=nu)
        nodes_x, weights_x, nodes_y, weights_y = space.nodes(extra=10)
        grid_x, grid_y = np.meshgrid(nodes_x, nodes_y)
        weights = np.outer(weights_y, weights_x)
        error = np.hypot(space.evaluate(solution.velocity_x, nodes_x, nodes_y) - exact_x(grid_x, grid_y),
                         space.evaluate(solution.velocity_y, nodes_x, nodes_y) - exact_y(grid_x, grid_y))
        errors_u.append(float(np.sqrt(np.sum(weights * error ** 2))))
        errors_p.append(float(np.sqrt(np.sum(weights * (space.evaluate_pressure(solution.pressure, nodes_x, nodes_y)
                                                        - exact_p(grid_x, grid_y)) ** 2))))
        assert solution.divergence_inf < 1e-3 * (1 + np.max(np.abs(exact_x(grid_x, grid_y))))
    # A smooth manufactured solution: four more modes per direction buy four decades here,
    # which is far beyond anything an algebraic rate of the same resolution could give.
    assert errors_u[1] < 1e-2 * errors_u[0], errors_u
    assert errors_u[2] < 1e-2 * errors_u[1], errors_u
    assert errors_u[3] < 1e-12, errors_u
    assert errors_p[3] < 1e-12, errors_p


def test_operators_are_a_compatible_pair_and_the_stiffness_form_is_symmetric():
    """The two identities the staggered MAC operators also satisfy, in weak form.

    D and G must be adjoints (`<G p, v> == -<p, D v>` with the mass inner product) or the
    saddle-point system is inconsistent, and the stiffness must be symmetric as a bilinear
    form, which in this representation means symmetric under the mass inner product. Both
    statements go through `inner` rather than a plain dot product, because the operators
    here are M^-1 S and M^-1 G: an unweighted dot product would test a different operator.
    """
    model = SpectralModel(Space(6, 1., 1.), .1)
    generator = np.random.default_rng(0)
    velocity = generator.standard_normal(2 * model.modes)
    other = generator.standard_normal(2 * model.modes)
    pressure = generator.standard_normal(model.modes)
    # `apply_D` drops the constant pressure row, which is the removed gauge, so the adjoint
    # identity is stated on the constrained rows. That mode of the pressure is zero anyway.
    assert abs(model.inner(model.apply_G(pressure), velocity)
               + pressure.reshape(-1)[1:] @ model.apply_D(velocity)) < 1e-11
    assert abs(model.inner(model.apply_K(velocity), other) - model.inner(velocity, model.apply_K(other))) < 1e-11


def _manufactured(lx, ly, nu, amplitude):
    """The transient manufactured solution the MAC temporal study uses, unit amplitude."""
    x, y, t = sy.symbols('x y t')
    stream = sy.sin(sy.pi * x / lx) ** 2 * sy.sin(sy.pi * y / ly) ** 2 * sy.exp(-t)
    u, v = sy.diff(stream, y), -sy.diff(stream, x)
    pressure = sy.cos(sy.pi * x / lx) * sy.cos(sy.pi * y / ly) * sy.exp(-t)
    forcing = []
    for q, axis in ((u, x), (v, y)):
        # Velocity scales with the amplitude, the convective term quadratically.
        forcing.append(amplitude * (-nu * (sy.diff(q, x, 2) + sy.diff(q, y, 2))
                                    + sy.diff(pressure, axis) + sy.diff(q, t))
                       + amplitude ** 2 * (u * sy.diff(q, x) + v * sy.diff(q, y)))
    functions = [sy.lambdify((x, y, t), expression, 'numpy') for expression in (forcing[0], forcing[1], u, v)]
    return functions


@pytest.mark.parametrize('scheme_name', ['sdirk2', 'sdirk2_mrsav'])
def test_the_schemes_drive_the_spectral_model_at_second_order_in_time(scheme_name):
    """The point of the seam: both schemes run on this discretisation unmodified.

    The reference is the same scheme at a much smaller step, so what is measured is the
    temporal error; the spatial error is far below it at this resolution. First order here
    means the nonlinear or the pressure term is wired in with the wrong mass convention,
    which is exactly the failure that a mesh-convergence check cannot see.
    """
    scheme = {'sdirk2': SDIRK2(), 'sdirk2_mrsav': SDIRK2MRSAV()}[scheme_name]
    lx = ly = 1.
    nu = .1
    amplitude = .1
    final_time = .2
    space = Space(16, lx, ly)
    forcing_u, forcing_v, exact_u, exact_v = _manufactured(lx, ly, nu, amplitude)
    model = SpectralModel(space, nu, tolerance=1e-12,
                          force=lambda t: np.concatenate([model.project(lambda x, y: forcing_u(x, y, t)),
                                                          model.project(lambda x, y: forcing_v(x, y, t))]))
    initial = model.state(0., np.concatenate([model.project(lambda x, y: amplitude * exact_u(x, y, 0.)),
                                              model.project(lambda x, y: amplitude * exact_v(x, y, 0.))]), 0.)

    # Check the physical initial field, not just a same-code time reference.
    nodes_x, weights_x, nodes_y, weights_y = space.nodes(extra=10)
    grid_x, grid_y = np.meshgrid(nodes_x, nodes_y)
    for coefficients, exact in ((initial.u, exact_u), (initial.v, exact_v)):
        np.testing.assert_allclose(space.evaluate(coefficients, nodes_x, nodes_y),
                                   amplitude * exact(grid_x, grid_y, 0.), rtol=0., atol=1e-11)

    def run(step):
        state = initial
        for _ in range(round(final_time / step)):
            state = scheme.step(model, state, step).state
        return model.vector(state)

    reference = run(final_time / 256)
    error = {}
    # Keep the original steps and ratio gate; compare the fine reference against the
    # analytic solution as well so a shared initial/forcing mismatch cannot pass silently.
    exact_error_squared = 0.
    for index, exact in enumerate((exact_u, exact_v)):
        coefficients = reference[index * model.modes:(index + 1) * model.modes].reshape(space.size, space.size)
        defect = space.evaluate(coefficients, nodes_x, nodes_y) - amplitude * exact(grid_x, grid_y, final_time)
        exact_error_squared += np.sum(np.outer(weights_y, weights_x) * defect ** 2)
    assert np.sqrt(exact_error_squared) < 1e-6, np.sqrt(exact_error_squared)
    for step in (final_time / 4, final_time / 8):
        difference = run(step) - reference
        error[step] = float(np.sqrt(max(model.inner(difference, difference), 0.)))
    coarse, medium = error[final_time / 4], error[final_time / 8]
    assert coarse > 100 * np.finfo(float).eps
    assert 3.4 < coarse / medium < 4.6, (coarse, medium, coarse / medium)


def _inhomogeneous_manufactured(lx, ly):
    """Exact divergence-free field whose wall trace matches a zero-mean lid profile.

    The profile is ``A(xi) = xi (1-xi^2)^2`` and the vertical factor h satisfies h'(1) =
    ly/2, so the top trace of u is exactly A at the reference coordinate. The profile is
    odd, which is what makes the net flux zero: with a nonzero flux the continuous problem
    has no divergence-free solution at all. That is the cavity's corner singularity rather
    than a property of the lifting, which is why the cavity is compared against the MAC
    solver instead of against an analytic field.
    """
    x, y, s, t = sy.symbols('x y s t')
    profile = s * (1 - s ** 2) ** 2
    vertical = -(ly * (t + 1) ** 2 * (1 - t)) / 8
    stream = profile.subs(s, 2 * x / lx - 1) * vertical.subs(t, 2 * y / ly - 1)
    u, v = sy.diff(stream, y), -sy.diff(stream, x)
    return profile, u, v


@pytest.mark.parametrize('pressure_kind', ['polynomial', 'smooth'])
def test_lifting_solves_an_inhomogeneous_manufactured_problem(pressure_kind):
    """The lifting, against an analytic solution that is *not* zero on the boundary.

    With a polynomial pressure the exact velocity minus the lifting lies in the velocity
    space, so the discrete solve has to reproduce it to roundoff; with a smooth pressure the
    pressure projection is the only error left and it has to be spectrally small. Without
    the lifting the interior unknown cannot satisfy a nonzero trace at all, so this fails
    before it converges rather than degrading slowly.
    """
    x, y, s = sy.symbols('x y s')
    lx = ly = 1.
    nu = .1
    profile, u, v = _inhomogeneous_manufactured(lx, ly)
    pressure = ((2 * x / lx - 1) * (2 * y / ly - 1) if pressure_kind == 'polynomial'
                else sy.cos(sy.pi * x / lx) * sy.cos(sy.pi * y / ly))
    forcing = [-nu * (sy.diff(q, x, 2) + sy.diff(q, y, 2)) + sy.diff(pressure, axis)
               for q, axis in ((u, x), (v, y))]
    exact_u, exact_v = sy.lambdify((x, y), u, 'numpy'), sy.lambdify((x, y), v, 'numpy')
    errors = []
    for size in (8, 12, 16):
        space = Space(size, lx, ly)
        lifting = LidLifting(space, sy.lambdify(s, profile, 'numpy'))
        model = SpectralModel(space, nu, lifting=lifting)
        solution = model.solve(np.concatenate([model.project(sy.lambdify((x, y), forcing[0], 'numpy')),
                                               model.project(sy.lambdify((x, y), forcing[1], 'numpy'))]),
                               mass=0., viscosity=nu)
        nodes_x, weights_x, nodes_y, weights_y = space.nodes(extra=10)
        grid_x, grid_y = np.meshgrid(nodes_x, nodes_y)
        error_u = space.evaluate(solution.velocity[:model.modes].reshape(size, size), nodes_x, nodes_y) \
            + lifting.velocity(nodes_x, nodes_y)[0] - exact_u(grid_x, grid_y)
        error_v = space.evaluate(solution.velocity[model.modes:].reshape(size, size), nodes_x, nodes_y) \
            + lifting.velocity(nodes_x, nodes_y)[1] - exact_v(grid_x, grid_y)
        errors.append(float(np.sqrt(np.sum(np.outer(weights_y, weights_x) * (error_u ** 2 + error_v ** 2)))))
    if pressure_kind == 'polynomial':
        # The exact solution minus the lifting lies in the space, so every size is at
        # roundoff and the sequence is not monotone; only the level is meaningful here.
        assert errors[2] < 1e-13, errors
    else:
        assert errors[2] < errors[1] < errors[0]
        assert errors[2] < 1e-9, errors


def _cell_centred(u, v):
    """Velocity at cell centres: u faces are already cell-centred in y and v in x."""
    return 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :])


def _streamfunction_from_vorticity(omega):
    """Solve -Delta psi=omega at unit-square cell centres, psi=0 on walls.

    Odd ghost values impose Dirichlet data at the wall half a cell away.
    The 1D positive Laplacian has boundary diagonal 3/h^2 (interior 2/h^2).
    Its orthonormal eigenvectors are DST-II modes; inverse is idstn(type=2).
    """
    n = omega.shape[0]
    k = np.arange(1, n+1)
    one = 4*n*n*np.sin(np.pi*k/(2*n))**2
    return idstn(dstn(omega, type=2, norm='ortho')/(one[:, None]+one[None, :]),
                 type=2, norm='ortho')


@pytest.mark.parametrize('size', [20])
def test_homogeneous_convection_has_zero_work_without_solenoidality(size):
    model = SpectralModel(Space(size, 1.3, .8), .1)
    velocity = np.random.default_rng(4).standard_normal(2 * model.modes)
    assert np.max(np.abs(model.apply_D(velocity))) > 1e-3
    nonlinear = model.nonlinear(velocity)
    scale = np.sqrt(model.inner(velocity, velocity) * model.inner(nonlinear, nonlinear))
    assert abs(model.inner(nonlinear, velocity)) < 1e-12 * scale


@pytest.mark.parametrize('size', [20])
@pytest.mark.parametrize('with_lifting', [True])
def test_dealiased_convection_matches_overintegrated_reference(size, with_lifting):
    space = Space(size, 1.3, .8)
    lifting = LidLifting(space) if with_lifting else None
    model = SpectralModel(space, .1, dealias=3, lifting=lifting)
    reference = SpectralModel(space, .1, dealias=2*size, lifting=lifting)
    assert 2*(size+model.quadrature_extra)-1 >= 3*(size+1)
    assert reference.quadrature_extra == 2*size
    velocity = np.random.default_rng(17).standard_normal(2*size*size)
    got, work = model.nonlinear_with_lifting(velocity)
    expected, expected_work = reference.nonlinear_with_lifting(velocity)
    difference = got-expected
    assert np.sqrt(model.inner(difference, difference)/model.inner(expected, expected)) < 1e-11
    assert abs(work-expected_work) < 1e-11*(1+abs(expected_work))
    assert abs(model.inner(got, velocity)+work) < 1e-11*(1+np.sqrt(model.inner(got, got)*model.inner(velocity, velocity)))




def test_stage_divergence_separates_physical_field_from_weak_constraint():
    """Underresolved velocity: both schemes must report its nonzero physical divergence."""
    model = SpectralModel(Space(4, 1., 1.), .1)
    force = np.random.default_rng(1).normal(size=2*model.modes)
    model.force = lambda t: force
    initial = model.state(0., model.zero_velocity())
    for scheme in (SDIRK2(), SDIRK2MRSAV()):
        trial = scheme.step(model, initial, .01)
        stage = trial.stages[-1]
        physical = model.diagnostics(trial.state)['divergence_inf']
        assert physical > .1
        assert stage.divergence_inf == pytest.approx(physical, rel=1e-12)
        assert stage.continuity_residual < 1e-12

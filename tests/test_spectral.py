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


@pytest.mark.parametrize('lx,ly', [(1., 1.), (1.3, .8)])
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


def test_quadrature_nodes_respect_each_direction():
    """Regression: an earlier version scaled the y nodes by lx, which left [-1, 1]."""
    space = Space(4, 1.3, .8)
    nodes_x, _, nodes_y, _ = space.nodes()
    assert nodes_x.min() > 0 and nodes_x.max() < 1.3
    assert nodes_y.min() > 0 and nodes_y.max() < .8


def test_load_of_a_basis_function_is_its_mass_column():
    """The right-hand side path, against a reference that needs no quadrature at all."""
    space = Space(5, 1.3, .8)
    mass = velocity_mass(space)
    for k, l in ((2, 3), (0, 0), (4, 1)):
        shape = np.zeros((space.size, space.size))
        shape[k, l] = 1.
        nodes_x, _, nodes_y, _ = space.nodes()
        load = space.load(lambda x, y: space.evaluate(shape, nodes_x, nodes_y))
        reference = mass @ shape.reshape(-1)
        assert np.max(np.abs(load.reshape(-1) - reference)) < 1e-11 * np.max(np.abs(reference))


def test_load_of_a_separable_function_factors_on_a_rectangle():
    """The rectangular reference that the transposed tensor-product order cannot pass.

    f = x * y(1 - y/ly) is separable, so the load must be the outer product of two 1D
    integrals, and its non-zero pattern {0,1} x {0,2} is not symmetric. The transposed
    order returns that outer product transposed, which passes every square-domain test
    (where the fields and basis are symmetric under x <-> y) and fails only here.
    """
    lx, ly = 1.3, .8
    space = Space(5, lx, ly)
    nodes_x, weights_x, nodes_y, weights_y = space.nodes(extra=6)
    basis_x = basis.dirichlet_values(space.size, 2 * nodes_x / lx - 1)
    basis_y = basis.dirichlet_values(space.size, 2 * nodes_y / ly - 1)
    integral_x = basis_x @ (nodes_x * weights_x)
    integral_y = basis_y @ ((nodes_y * (1 - nodes_y / ly)) * weights_y)
    load = space.load(lambda x, y: x * (y * (1 - y / ly)))
    np.testing.assert_allclose(load, np.outer(integral_x, integral_y), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('lx,ly', [(1., 1.), (1.3, .8)])
def test_diffusion_reproduces_a_polynomial_exactly(lx, ly):
    """The strongest per-mode reference: no pressure coupling and the exact solution is in
    the space, so the discrete solution must be exact up to roundoff. This is the test that
    exposed the transposed load, which every square-domain check passed."""
    import sympy as sy
    from scipy.sparse.linalg import splu
    symbolic_x, symbolic_y = sy.symbols('x y')
    nu = .1
    exact = (symbolic_x / lx) * (1 - symbolic_x / lx) * (symbolic_y / ly) * (1 - symbolic_y / ly)
    forcing = -nu * (sy.diff(exact, symbolic_x, 2) + sy.diff(exact, symbolic_y, 2))
    exact_function = sy.lambdify((symbolic_x, symbolic_y), exact, 'numpy')
    forcing_function = sy.lambdify((symbolic_x, symbolic_y), forcing, 'numpy')
    for size in (3, 6, 10):
        space = Space(size, lx, ly)
        coefficients = splu((nu * velocity_stiffness(space)).tocsc()).solve(space.load(forcing_function).reshape(-1))
        nodes_x, _, nodes_y, _ = space.nodes(extra=8)
        grid_x, grid_y = np.meshgrid(nodes_x, nodes_y)
        error = space.evaluate(coefficients.reshape(size, size), nodes_x, nodes_y) - exact_function(grid_x, grid_y)
        assert np.max(np.abs(error)) < 1e-13


@pytest.mark.parametrize('lx,ly', [(1., 1.), (1.3, .8)])
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


def _coefficients(space, mass, expression):
    """Coefficients of a polynomial expression that lies in the velocity space.

    `Space.load` is the weak right-hand side, i.e. the mass matrix applied to the
    coefficients, so it has to be solved before it can be used as a field. Confusing the
    two is worth 30% on a smooth field, not roundoff, so this path is spelled out.
    """
    from scipy.sparse.linalg import splu
    grid_load = space.load(sy.lambdify(('x', 'y'), expression, 'numpy')).reshape(-1)
    return splu(mass.tocsc()).solve(grid_load)


def test_nonlinear_term_matches_analytic_convection():
    """N(v) = (u.grad)u against sympy, on a polynomial velocity that is in the space.

    The stream function is chosen so that both velocity components vanish on all four
    walls: the model imposes no-slip on both components, so a field built from a generic
    stream function is not in the space and would only be tested through its projection.
    The quadrature is wide enough that the projection of this product is exact, which is
    what makes a pointwise comparison against the analytic value meaningful.
    """
    x, y = sy.symbols('x y')
    stream = x ** 2 * (1 - x) ** 2 * y ** 2 * (1 - y) ** 2
    u, v = sy.diff(stream, y), -sy.diff(stream, x)
    convection_u = sy.diff(u, x) * u + sy.diff(u, y) * v
    convection_v = sy.diff(v, x) * u + sy.diff(v, y) * v
    space = Space(8, 1., 1.)
    model = SpectralModel(space, .1)
    velocity = np.concatenate([_coefficients(space, model.mass, u), _coefficients(space, model.mass, v)])
    assembled = model.nonlinear(velocity)
    nodes_x, _, nodes_y, _ = space.nodes(extra=10)
    grid_x, grid_y = np.meshgrid(nodes_x, nodes_y)
    for component, expression in ((assembled[:model.modes], convection_u),
                                  (assembled[model.modes:], convection_v)):
        values = space.evaluate(component.reshape(space.size, space.size), nodes_x, nodes_y)
        reference = sy.lambdify((x, y), expression, 'numpy')(grid_x, grid_y)
        assert np.max(np.abs(values - reference)) < 1e-13
        assert np.max(np.abs(reference)) > 1e-4      # the check has something to measure


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


def test_projection_returns_coefficients_not_the_right_hand_side():
    """`project` must solve the mass matrix: a field in the space is reproduced exactly.

    `Space.load` is only the weak right-hand side. Treating it as a field is worth an O(1)
    error on smooth data, and it is the same mistake that made `nonlinear` look wrong.
    """
    x, y = sy.symbols('x y')
    stream = x ** 2 * (1 - x) ** 2 * y ** 2 * (1 - y) ** 2
    space = Space(8, 1., 1.)
    model = SpectralModel(space, .1)
    coefficients = model.project(sy.lambdify((x, y), sy.diff(stream, y), 'numpy'))
    nodes_x, _, nodes_y, _ = space.nodes(extra=10)
    grid_x, grid_y = np.meshgrid(nodes_x, nodes_y)
    values = space.evaluate(coefficients.reshape(space.size, space.size), nodes_x, nodes_y)
    assert np.max(np.abs(values - sy.lambdify((x, y), sy.diff(stream, y), 'numpy')(grid_x, grid_y))) < 1e-13


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


def test_the_lifting_trace_is_the_prescribed_lid():
    """The boundary values are carried by the lifting, so its trace *is* the boundary condition.

    Checked on all four walls for the regularised profile, which the 1D Dirichlet basis
    represents exactly; the sharp lid is only reproduced as its projection, which is what
    the cavity comparison documents.
    """
    space = Space(10, 1., 1.)
    lifting = LidLifting(space, lambda ξ: (1 - np.asarray(ξ) ** 2) ** 2)
    x = np.linspace(0, 1, 33)
    lid, _ = lifting.velocity(x, np.full_like(x, 1.))
    assert np.max(np.abs(lid - (1 - (2 * x - 1) ** 2) ** 2)) < 1e-14
    bottom, _ = lifting.velocity(x, np.zeros_like(x))
    assert np.max(np.abs(bottom)) < 1e-14
    y = np.linspace(0, 1, 33)
    for side in (0., 1.):
        values, cross = lifting.velocity(np.full_like(y, side), y)
        assert np.max(np.abs(values)) < 1e-14
        assert np.max(np.abs(cross)) < 1e-14


def _mac_cavity(n, nu, speed, profile, final_time, dt):
    """Run the validated MAC cavity, whose lid enters as a ghost-point viscous load."""
    from solver.mac.grid import MACGrid
    from solver.mac_ns import MACNavierStokes
    from experiments.cavity.model import lid_viscous_load
    grid = MACGrid(n, n, 1., 1.)
    model = MACNavierStokes(grid, nu, force=lambda t: lid_viscous_load(grid, nu, speed, profile))
    scheme = SDIRK2MRSAV()
    state = model.state(0., np.zeros(grid.size), 0.)
    for _ in range(round(final_time / dt)):
        state = scheme.step(model, state, dt).state
    u, v = grid.unpack(model.vector(state))
    return grid, u, v


def _spectral_cavity(size, nu, profile, final_time, dt):
    """Run the same cavity on the spectral model, whose lid enters as a lifting."""
    space = Space(size, 1., 1.)
    lifting = LidLifting(space) if profile is None else LidLifting(space, profile)
    model = SpectralModel(space, nu, lifting=lifting)
    scheme = SDIRK2MRSAV()
    state = model.state(0., model.zero_velocity(), 0.)
    for _ in range(round(final_time / dt)):
        state = scheme.step(model, state, dt).state
    return space, lifting, state


@pytest.mark.parametrize('lid,size', [('sharp', 20), ('regularised', 16)])
def test_spectral_cavity_agrees_with_the_validated_mac_lid_load(lid, size):
    """S3's comparison: two discretisations, one scheme, one benchmark flow.

    The yardstick is the MAC solver's own grid sensitivity, measured here rather than
    assumed: the spectral solution has to be at least as close to the fine MAC solution as
    the coarse MAC grid is. The lid is imposed completely differently on the two sides -- a
    ghost-point viscous load on the MAC, a lifting whose trace is the wall value in the
    spectral space -- so agreeing on the primary vortex at Re=100 is a real statement about
    both. The sharp lid is the harder case because its constant profile is not representable
    and arrives as a projection with Gibbs oscillations near the corners.
    """
    from experiments.cavity.model import regularised_lid
    nu = .01
    speed = 1.
    final_time = 10.
    dt = .04
    profile = regularised_lid if lid == 'regularised' else None
    coarse_grid, coarse_u, coarse_v = _mac_cavity(32, nu, speed, profile, final_time, dt)
    grid, u, v = _mac_cavity(64, nu, speed, profile, final_time, dt)
    area = grid.hx * grid.hy
    difference_u = coarse_u[1:-1, 1:-1] - u[1::2, ::2][1:-1, 1:-1]
    difference_v = coarse_v[1:-1, 1:-1] - v[::2, 1::2][1:-1, 1:-1]
    yardstick = float(np.sqrt((np.sum(difference_u ** 2) + np.sum(difference_v ** 2)) * area))
    space, lifting, state = _spectral_cavity(size, nu, profile, final_time, dt)
    # The grid is a tensor product, so the axes of the meshgrids are the point lists the
    # spectral basis expects; flattened meshgrids would build a product, not a pairing.
    xu, yu = grid.coordinates('u')
    xv, yv = grid.coordinates('v')
    spectral_u = space.evaluate(state.u, xu[0, :], yu[:, 0]) + lifting.velocity(xu[0, :], yu[:, 0])[0]
    spectral_v = space.evaluate(state.v, xv[0, :], yv[:, 0]) + lifting.velocity(xv[0, :], yv[:, 0])[1]
    error = float(np.sqrt((np.sum((spectral_u[1:-1, 1:-1] - u[1:-1, 1:-1]) ** 2)
                           + np.sum((spectral_v[1:-1, 1:-1] - v[1:-1, 1:-1]) ** 2)) * area))
    assert error <= yardstick, (error, yardstick)


@pytest.mark.parametrize('size', [8, 20, 28])
def test_homogeneous_convection_has_zero_work_without_solenoidality(size):
    model = SpectralModel(Space(size, 1.3, .8), .1)
    velocity = np.random.default_rng(4).standard_normal(2 * model.modes)
    assert np.max(np.abs(model.apply_D(velocity))) > 1e-3
    nonlinear = model.nonlinear(velocity)
    scale = np.sqrt(model.inner(velocity, velocity) * model.inner(nonlinear, nonlinear))
    assert abs(model.inner(nonlinear, velocity)) < 1e-12 * scale


def test_homogeneous_skew_convection_matches_independent_polynomial_load():
    # A non-solenoidal field checks the half-divergence correction, which the
    # existing divergence-free analytic test cannot distinguish from advection.
    x, y = sy.symbols('x y')
    u = x * (1.3 - x) * y * (.8 - y)
    v = x * u
    divergence = sy.diff(u, x) + sy.diff(v, y)
    model = SpectralModel(Space(8, 1.3, .8), .1)
    velocity = np.concatenate([_coefficients(model.space, model.mass, q) for q in (u, v)])
    got = model.nonlinear(velocity)
    for index, q in enumerate((u, v)):
        exact = u * sy.diff(q, x) + v * sy.diff(q, y) + divergence * q / 2
        load = model.space.load(sy.lambdify((x, y), exact, 'numpy'), extra=20).reshape(-1)
        actual = model.mass @ got[index * model.modes:(index + 1) * model.modes]
        np.testing.assert_allclose(actual, load, rtol=1e-11, atol=1e-13)


def _total_skew_pairing(model, source, target):
    """Independent physical-space b(source+g,source+g,target+g)."""
    space = model.space
    x, wx, y, wy = space.nodes(extra=model.quadrature_extra)
    px, py = space.velocity_values(x, y)
    dx, dy = space.velocity_derivatives(x, y)

    def samples(vector):
        a, b = vector.reshape(2, space.size, space.size)
        u, v = space.evaluate(a, x, y), space.evaluate(b, x, y)
        ux, uy, vx, vy = py.T @ a.T @ dx, dy.T @ a.T @ px, py.T @ b.T @ dx, dy.T @ b.T @ px
        if model.lifting is not None:
            gu, gv = model.lifting.velocity(x, y)
            gux, guy, gvx, gvy = model.lifting.derivatives(x, y)
            u, v, ux, uy, vx, vy = u+gu, v+gv, ux+gux, uy+guy, vx+gvx, vy+gvy
        return u, v, ux, uy, vx, vy

    u, v, ux, uy, vx, vy = samples(source)
    a, b, ax, ay, bx, by = samples(target)
    value = (u*ux+v*uy)*a + (u*vx+v*vy)*b - (u*ax+v*ay)*u - (u*bx+v*by)*v
    return .5 * float(np.sum(np.outer(wy, wx) * value))


def test_lifting_convection_pairs_with_total_velocity():
    space = Space(8, 1.3, .8)
    model = SpectralModel(space, .1, lifting=LidLifting(space, lambda x: (1-x*x)**2))
    rng = np.random.default_rng(41)
    source, target = rng.standard_normal((2, 2*model.modes))
    convection, work = model.nonlinear_with_lifting(source)
    assert abs(work) > 1e-3  # omission of the lifting work is observable
    assert abs(model.inner(convection, source)+work) < 1e-11
    np.testing.assert_allclose(model.inner(convection, target)+work,
                               _total_skew_pairing(model, source, target), rtol=1e-12, atol=1e-11)


def test_zero_lifting_reproduces_homogeneous_sav_step():
    space = Space(8)
    homogeneous = SpectralModel(space, .1)
    lifted = SpectralModel(space, .1, lifting=LidLifting(space, lambda x: np.zeros_like(x)))
    velocity = homogeneous.solve(np.random.default_rng(2).standard_normal(128), mass=1., viscosity=.01).velocity
    first = SDIRK2MRSAV().step(homogeneous, homogeneous.state(0., velocity), .001)
    second = SDIRK2MRSAV().step(lifted, lifted.state(0., velocity), .001)
    np.testing.assert_allclose(lifted.vector(second.state), homogeneous.vector(first.state), rtol=1e-12, atol=1e-13)
    assert abs(first.state.r-second.state.r) < 1e-13


def test_lifting_sav_stages_satisfy_total_velocity_scalar_equations():
    from solver.schemes.sdirk2 import ETA, DELTA

    class RecordedScheme(SDIRK2MRSAV):
        def _stage(self, *args, **kwargs):
            result = super()._stage(*args, **kwargs)
            self.velocities.append(result[0].copy())
            return result

    space = Space(8)
    model = SpectralModel(space, .1, lifting=LidLifting(space, lambda x: (1-x*x)**2))
    # Start on the affine divergence constraint; zero interior coefficients would not.
    old = model.solve(model.zero_velocity(), mass=1., viscosity=.01).velocity
    scheme = RecordedScheme(); scheme.velocities = []
    dt = .003
    trial = scheme.step(model, model.state(0., old, 0.), dt)
    first, second = scheme.velocities
    r1, r2 = (stage.r for stage in trial.stages)
    pairing1 = ETA * _total_skew_pairing(model, old, first)
    pairing2 = -_total_skew_pairing(model, old, second)+(1-DELTA)*_total_skew_pairing(model, first, second)
    residual1 = (1+scheme.gamma*ETA*dt)*r1 + dt*(1+r1)*pairing1
    residual2 = ((1+scheme.gamma*ETA*dt)*r2-(1-scheme.gamma*dt*(1-2*ETA))*r1
                 + dt*(1+r2)*pairing2)
    assert max(abs(residual1), abs(residual2)) < 1e-12
    assert max(stage.residual for stage in trial.stages) < 1e-8


def test_regularised_lifting_sav_time_refinement():
    from solver.spectral.lifting import regularised
    space = Space(12)
    model = SpectralModel(space, .01, lifting=LidLifting(space, regularised))
    initial = model.solve(model.zero_velocity(), mass=1., viscosity=.01).velocity
    velocities, peaks = [], []
    for dt in (.02, .01, .005, .0025, .00125):
        state = model.state(0., initial.copy(), 0.)
        peak = 0.
        for _ in range(round(.2/dt)):
            trial = SDIRK2MRSAV().step(model, state, dt)
            state = trial.state
            peak = max(peak, *(abs(stage.r) for stage in trial.stages))
        velocities.append(model.vector(state))
        peaks.append(peak)
    # A fixed-space regression, not a claim of a general time-uniform bound.
    ratios = np.array(peaks[:-1]) / np.array(peaks[1:])
    assert np.all((ratios > 1.8) & (ratios < 2.2)), peaks
    errors = [np.sqrt(model.inner(v-velocities[-1], v-velocities[-1])) for v in velocities[:3]]
    assert 3.3 < errors[0]/errors[1] < 5., errors
    assert 3.3 < errors[1]/errors[2] < 5., errors


@pytest.mark.parametrize('size', [20, 28, 40])
@pytest.mark.parametrize('with_lifting', [False, True])
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

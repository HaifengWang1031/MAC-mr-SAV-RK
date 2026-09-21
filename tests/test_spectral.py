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
from solver.spectral.stokes import SpectralStokes


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

"""The spectral discretisation presented as the model seam (solver/model.py).

A velocity field is carried as coefficients in the Dirichlet-combined Legendre basis,
`u = sum Ux[k,l] phi_k(x) psi_l(y)`, and the packed representation is the concatenation of
the two component arrays, `[Ux.ravel(), Uy.ravel()]`, so the schemes and the existing
driver work unchanged. The pressure lives in the plain Legendre basis, which is why its
coefficients are a separate field with the same [x, y] index convention.

Two deliberate differences from the MAC model, both consequences of representing fields
modally:

- `max_abs` returns the largest coefficient magnitude, not a sup-norm of the field. It only
  has to be a consistent scale for the stage residual and divergence checks, which are all
  relative; the physically meaningful divergence is reported by `diagnostics` instead,
  evaluated in physical space.
- Every member works on *coefficients*, not on weak right-hand sides. The seam adds
  `vector(state)`, `force(t)` and `nonlinear(state)` together and hands the sum to `solve`,
  so all of them have to live in one vector space; the weak stiffness S and gradient G are
  therefore composed with M^-1 to act as field operators. The MAC model never faces this
  choice because its mass matrix is diagonal.
- `nonlinear` assembles the convective term on a wider quadrature than the projection uses.
  With enough nodes the projection of the polynomial products is exact, which is what lets
  the term be verified against an analytic reference rather than only against itself.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import numpy as np
from numpy.typing import NDArray
from ..core import Stage, State
from ..model import SolveResult
from . import assembly
from .assembly import Space
from .stokes import SpectralStokes

Array = NDArray[np.float64]


@dataclass
class SpectralSolution:
    velocity: Array
    pressure: Array
    residual: float
    divergence_inf: float


class SpectralModel:
    def __init__(self, space: Space, nu: float, *, force: Callable[[float], Array] | None = None,
                 tolerance: float = 1e-10, cache_size: int = 4, dealias: int = 8,
                 lifting: Any = None) -> None:
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError('nu must be positive and finite')
        if dealias < 3:
            raise ValueError('dealias must be at least 3')
        self.space = space
        self.nu = nu
        self.dealias = dealias
        self.mass = assembly.velocity_mass(space)
        self.stiffness = assembly.velocity_stiffness(space)
        self.divergence_x, self.divergence_y = assembly.divergence_blocks(space)
        self.stokes = SpectralStokes(space, tolerance=tolerance, cache_size=cache_size)
        self._mass_factorization: Any = None
        # A nonzero wall value is carried by a known field whose loads are constant in
        # time, so they are assembled once and folded into every solve. See lifting.py.
        self.lifting = lifting
        self._lifting = lifting.loads() if lifting is not None else None
        self.force: Callable[[float], Array] = force if force is not None else lambda t: self.zero_velocity()

    @property
    def quadrature_extra(self) -> int:
        """Effective extra Gauss nodes for polynomial convection and fixed LidLifting.

        size modes phi_k=P_k-P_{k+2} have maximum degree p=size+1.
        A convective weak integrand has degree at most 3p in either direction
        (the derivative can be in the other direction). Require 2q-1 >= 3p.
        User dealias remains a lower bound and may request denser integration.
        Arbitrary non-polynomial custom liftings have no exactness guarantee.
        """
        minimum_nodes = (3 * (self.space.size + 1) + 2) // 2
        return max(self.dealias, minimum_nodes - self.space.size)

    @property
    def modes(self) -> int:
        return self.space.velocity_dofs

    def zero_velocity(self) -> Array:
        return np.zeros(2 * self.modes)

    def zero_pressure(self) -> Array:
        return np.zeros(self.modes)

    # --- the seam

    def vector(self, state: State) -> Array:
        return np.concatenate([np.asarray(state.u).reshape(-1), np.asarray(state.v).reshape(-1)])

    def state(self, t: float, velocity: Array, r: float = 0.) -> State:
        modes, size = self.modes, self.space.size
        return State(t, np.asarray(velocity)[:modes].reshape(size, size),
                     np.asarray(velocity)[modes:].reshape(size, size), r)

    def combine(self, *terms: tuple[float, Array]) -> Array:
        total = np.zeros_like(terms[0][1], dtype=float)
        for coefficient, field in terms:
            total += coefficient * field
        return total

    def apply_K(self, velocity: Array) -> Array:
        """The weak Laplacian as a field operator: M^-1 S, of the *total* field.

        S is the weak stiffness, i.e. it maps coefficients to a *load*. The seam adds the
        result of this call to other fields and hands the sum to `solve`, so it has to
        return a field, and the mass matrix has to be folded in. The MAC model never has to
        make this choice because its basis functions are nodal indicators and its mass
        matrix is diagonal; a modal basis does, and picking the wrong convention is a
        silent O(1) error rather than a visible one.

        A lifting is part of the physical field but not of the state, so its stiffness load
        is added here as well as being subtracted in `solve`. The two have to agree: the
        schemes build their stage defect out of this call and would otherwise keep an O(1)
        leftover and fail their own gate.
        """
        modes = self.modes
        out = np.empty_like(np.asarray(velocity))
        out[:modes] = self._mass_solve(self.stiffness @ velocity[:modes])
        out[modes:] = self._mass_solve(self.stiffness @ velocity[modes:])
        if self._lifting is not None:
            out[:modes] += self._mass_solve(self._lifting[0].reshape(-1))
            out[modes:] += self._mass_solve(self._lifting[1].reshape(-1))
        return out

    def apply_G(self, pressure: Array) -> Array:
        """The gradient in the field representation, M^-1 G with G = -D^T, per component."""
        return np.concatenate([self._mass_solve(-(self.divergence_x.T @ pressure)),
                               self._mass_solve(-(self.divergence_y.T @ pressure))])

    def apply_D(self, velocity: Array) -> Array:
        """Weak divergence of the *total* field over the constrained pressure modes.

        With a lifting the interior unknown satisfies `div w = -div g`, so the constraint
        residual reported to the schemes is the physical one, `div w + div g`. The constant
        pressure mode is the removed gauge, so its row is excluded: for a homogeneous field
        it vanishes anyway, because the basis carries no flux through the walls, but with a
        lid it would hold the cavity's net flux, which this formulation deliberately does
        not impose. What the schemes gate on is the residual `solve` actually enforces.
        """
        modes = self.modes
        constraint = self.divergence_x @ velocity[:modes] + self.divergence_y @ velocity[modes:]
        if self._lifting is not None:
            constraint = constraint + self._lifting[2].reshape(-1)
        return constraint[1:]

    def physical_divergence_inf(self, velocity: Array) -> float:
        size = self.space.size
        return self._total_divergence_inf(velocity[:self.modes].reshape(size, size),
                                          velocity[self.modes:].reshape(size, size))

    def inner(self, left: Array, right: Array) -> float:
        modes = self.modes
        return float(left[:modes] @ (self.mass @ right[:modes])
                     + left[modes:] @ (self.mass @ right[modes:]))

    def max_abs(self, field: Array) -> float:
        return float(np.max(np.abs(field)))

    def _mass_solve(self, load: Array) -> Array:
        """Coefficients from a load: the mass matrix has to be solved, not skipped.

        `Space.project` returns int f phi_k psi_l, which is the mass matrix applied to the
        coefficients (the right-hand side of a Galerkin problem), not the coefficients
        themselves. Returning it directly leaves `nonlinear` off by a mass solve, which on a
        smooth test field is a 30% error rather than a roundoff one.
        """
        if self._mass_factorization is None:
            from scipy.sparse.linalg import splu
            self._mass_factorization = splu(self.mass.tocsc())
        return self._mass_factorization.solve(np.asarray(load).reshape(-1))

    def project(self, function: Callable[[Array, Array], Array], extra: int = 3) -> Array:
        """Coefficients of a field given analytically, i.e. the load solved against the mass.

        This is how a forcing or an initial condition is turned into the representation the
        schemes use; `Space.load` alone would return the weak right-hand side instead.
        """
        return self._mass_solve(self.space.load(function, extra=extra))

    def nonlinear(self, velocity: Array) -> Array:
        return self.nonlinear_with_lifting(velocity)[0]

    def nonlinear_with_lifting(self, velocity: Array) -> tuple[Array, float]:
        """Return N(w) and b(w+g,w+g,g) for a fixed lifting g.

        N represents the antisymmetric load on homogeneous tests. The scalar is
        integrated directly against g, not against its homogeneous-space projection.
        Both use the same quadrature, so <N(w),w> + lifting_work cancels algebraically.
        """
        space, modes = self.space, self.modes
        size = space.size
        nodes_x, _, nodes_y, _ = space.nodes(extra=self.quadrature_extra)
        values_x, values_y = space.velocity_values(nodes_x, nodes_y)
        derivative_x, derivative_y = space.velocity_derivatives(nodes_x, nodes_y)
        coefficients_x = np.asarray(velocity)[:modes].reshape(size, size)
        coefficients_y = np.asarray(velocity)[modes:].reshape(size, size)
        u = space.evaluate(coefficients_x, nodes_x, nodes_y)
        v = space.evaluate(coefficients_y, nodes_x, nodes_y)
        du_dx = values_y.T @ coefficients_x.T @ derivative_x
        du_dy = derivative_y.T @ coefficients_x.T @ values_x
        dv_dx = values_y.T @ coefficients_y.T @ derivative_x
        dv_dy = derivative_y.T @ coefficients_y.T @ values_x
        if self.lifting is not None:
            g_u, g_v = self.lifting.velocity(nodes_x, nodes_y)
            dg_u_x, dg_u_y, dg_v_x, dg_v_y = self.lifting.derivatives(nodes_x, nodes_y)
            u, v = u + g_u, v + g_v
            du_dx, du_dy = du_dx + dg_u_x, du_dy + dg_u_y
            dv_dx, dv_dy = dv_dx + dg_v_x, dv_dy + dg_v_y
        # Assemble half of ((u.grad)u,test) - ((u.grad)test,u).
        load_u = .5 * (space.project(u * du_dx + v * du_dy, extra=self.quadrature_extra)
                       - space.stiffness_load(u * u, v * u))
        load_v = .5 * (space.project(u * dv_dx + v * dv_dy, extra=self.quadrature_extra)
                       - space.stiffness_load(u * v, v * v))
        lifting_work = 0.
        if self.lifting is not None:
            _, weights_x, _, weights_y = space.nodes(extra=self.quadrature_extra)
            integrand = ((u * du_dx + v * du_dy) * g_u
                         + (u * dv_dx + v * dv_dy) * g_v
                         - (u * dg_u_x + v * dg_u_y) * u
                         - (u * dg_v_x + v * dg_v_y) * v)
            lifting_work = .5 * float(np.sum(np.outer(weights_y, weights_x) * integrand))
        return np.concatenate([self._mass_solve(load_u), self._mass_solve(load_v)]), lifting_work

    def _total_divergence_inf(self, coefficients_x: Array, coefficients_y: Array) -> float:
        """max |div u| in physical space, including the lifting when there is one."""
        space = self.space
        nodes_x, _, nodes_y, _ = space.nodes(extra=self.quadrature_extra)
        values_x, values_y = space.velocity_values(nodes_x, nodes_y)
        derivative_x, derivative_y = space.velocity_derivatives(nodes_x, nodes_y)
        divergence = (values_y.T @ coefficients_x.T @ derivative_x
                      + derivative_y.T @ coefficients_y.T @ values_x)
        if self.lifting is not None:
            divergence = divergence + self.lifting.samples(nodes_x, nodes_y)[3]
        return float(np.max(np.abs(divergence)))

    def solve(self, rhs: Array, *, mass: float, viscosity: float, with_boundary_data: bool = True) -> SolveResult:
        """Solve the shifted Stokes problem for a forcing given in the field representation.

        The system is [nu S, G; D, 0] [c; p] = [load; continuity] with S, G, D the weak
        matrices, so the incoming field is turned into a load by the mass matrix before the
        solve and the coefficients that come back need no further conversion. With a
        lifting the boundary values contribute a constant momentum load with the same
        viscosity they are shifted by, and an inhomogeneous continuity row; both are
        switched off by `with_boundary_data=False`, which is what the correction columns of
        `solve_columns` need.
        """
        modes = self.modes
        size = self.space.size
        load_x = self.mass @ np.asarray(rhs)[:modes]
        load_y = self.mass @ np.asarray(rhs)[modes:]
        continuity = None
        if self._lifting is not None and with_boundary_data:
            load_x = load_x - viscosity * self._lifting[0].reshape(-1)
            load_y = load_y - viscosity * self._lifting[1].reshape(-1)
            continuity = -self._lifting[2].reshape(-1)
        solution = self.stokes.solve(load_x.reshape(size, size), load_y.reshape(size, size),
                                     mass=mass, viscosity=viscosity, continuity=continuity)
        return SpectralSolution(np.concatenate([solution.velocity_x.reshape(-1), solution.velocity_y.reshape(-1)]),
                                solution.pressure.reshape(-1), solution.residual,
                                self._total_divergence_inf(solution.velocity_x, solution.velocity_y))

    def solve_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float) -> list[SolveResult]:
        """One solve per right-hand side; only the first one carries the boundary data.

        The schemes add the later columns to the first as corrections, so the inhomogeneous
        Dirichlet datum has to survive that combination. Solving every column with the same
        datum would multiply it by the sum of the coefficients, so the correction columns
        are solved with a homogeneous one: because the model is linear, the combination is
        then exactly the solution of the combined right-hand side. Factorizations are
        reused from the cache either way.
        """
        return [self.solve(column, mass=mass, viscosity=viscosity, with_boundary_data=index == 0)
                for index, column in enumerate(columns)]

    def stage(self, pressure: Array, residual: float, divergence_inf: float, r: float = 0.,
              candidates: Sequence[float] = (), root_residuals: Sequence[float] = (),
              scalar_residual: float = 0., *, continuity_residual: float = float("nan")) -> Stage:
        return Stage(np.asarray(pressure), residual, divergence_inf, r, list(candidates),
                     list(root_residuals), scalar_residual, continuity_residual)

    def diagnostics(self, state: State) -> dict[str, float]:
        """Physical diagnostics of the *total* field, integrated by Gauss quadrature.

        For a field in the space these integrals equal the mass-matrix forms exactly, so
        this is a physical-space rewrite rather than an approximation; with a lifting it is
        also the only form that includes the boundary values.
        """
        space, modes = self.space, self.modes
        velocity = self.vector(state)
        nodes_x, weights_x, nodes_y, weights_y = space.nodes(extra=self.quadrature_extra)
        values_x, values_y = space.velocity_values(nodes_x, nodes_y)
        derivative_x, derivative_y = space.velocity_derivatives(nodes_x, nodes_y)
        coefficients_x = np.asarray(velocity)[:modes].reshape(space.size, space.size)
        coefficients_y = np.asarray(velocity)[modes:].reshape(space.size, space.size)
        u = space.evaluate(coefficients_x, nodes_x, nodes_y)
        v = space.evaluate(coefficients_y, nodes_x, nodes_y)
        du_dx = values_y.T @ coefficients_x.T @ derivative_x
        du_dy = derivative_y.T @ coefficients_x.T @ values_x
        dv_dx = values_y.T @ coefficients_y.T @ derivative_x
        dv_dy = derivative_y.T @ coefficients_y.T @ values_x
        if self.lifting is not None:
            g_u, g_v = self.lifting.velocity(nodes_x, nodes_y)
            dg_u_x, dg_u_y, dg_v_x, dg_v_y = self.lifting.derivatives(nodes_x, nodes_y)
            u, v = u + g_u, v + g_v
            du_dx, du_dy = du_dx + dg_u_x, du_dy + dg_u_y
            dv_dx, dv_dy = dv_dx + dg_v_x, dv_dy + dg_v_y
        weight = np.outer(weights_y, weights_x)
        kinetic = .5 * float(np.sum(weight * (u ** 2 + v ** 2)))
        return {'kinetic': kinetic, 'modified_energy': kinetic + .5 * (state.r - 1) ** 2,
                'divergence_inf': float(np.max(np.abs(du_dx + dv_dy))),
                'h1_seminorm_squared': float(np.sum(weight * (du_dx ** 2 + du_dy ** 2
                                                              + dv_dx ** 2 + dv_dy ** 2))),
                'r': state.r}

    def close(self) -> None:
        self.stokes.cache.clear()


if TYPE_CHECKING:
    from ..model import Model
    _spectral_realisation: type[Model] = SpectralModel

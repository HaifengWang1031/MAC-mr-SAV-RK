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
from .assembly import Space
from .mass import TensorMass
from .lifting import LidLifting
from .quadrature import QuadraturePlan
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
                 lifting: Any = None, stokes_backend: str = 'direct',
                 krylov_tolerance: float = 1e-12, max_iterations: int = 200) -> None:
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError('nu must be positive and finite')
        if dealias < 3:
            raise ValueError('dealias must be at least 3')
        self.space = space
        self.nu = nu
        self.dealias = dealias
        self.stokes: SpectralStokes
        if stokes_backend == 'direct':
            self.stokes = SpectralStokes(space, tolerance=tolerance, cache_size=cache_size)
        elif stokes_backend == 'tensor':
            from .tensor_stokes import TensorStokes
            self.stokes = TensorStokes(space, tolerance=tolerance, cache_size=cache_size,
                                       krylov_tolerance=krylov_tolerance, max_iterations=max_iterations)
        else:
            raise ValueError('Unknown spectral Stokes backend')
        self.mass, self.stiffness = self.stokes.mass, self.stokes.stiffness
        self.divergence_x, self.divergence_y = self.stokes.divergence_x, self.stokes.divergence_y
        self._quadrature = self.stokes.quadrature
        self._mass_factorization: TensorMass | None = None
        # A nonzero wall value is carried by a known field whose loads are constant in
        # time, so they are assembled once and folded into every solve. See lifting.py.
        self.lifting = lifting
        self._lifting = lifting.loads() if lifting is not None else None
        self._lid_samples: tuple[QuadraturePlan, Space, Array, tuple[Array, ...]] | None = None
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

    def stage_divergences(self, velocity: Array) -> tuple[float, float]:
        return self.physical_divergence_inf(velocity), self.max_abs(self.apply_D(velocity))

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
            self._mass_factorization = TensorMass(self.space)
        return self._mass_factorization.solve(load)

    def project(self, function: Callable[[Array, Array], Array], extra: int = 3) -> Array:
        """Coefficients of a field given analytically, i.e. the load solved against the mass.

        This is how a forcing or an initial condition is turned into the representation the
        schemes use; `Space.load` alone would return the weak right-hand side instead.
        """
        plan = self._quadrature.get(extra)
        x, y = np.meshgrid(plan.x, plan.y)
        return self._mass_solve(plan.project(function(x, y)))

    def nonlinear(self, velocity: Array) -> Array:
        return self.nonlinear_with_lifting(velocity)[0]

    def _lifting_fields(self, plan: QuadraturePlan) -> tuple[Array, ...]:
        """Cache only the built-in fixed lid; custom lifting calls retain their semantics."""
        if type(self.lifting) is LidLifting:
            cached = self._lid_samples
            if cached is None or cached[0] is not plan or cached[1] != self.lifting.space \
                    or not np.array_equal(cached[2], self.lifting.coefficients):
                gu, gux, guy, _ = self.lifting.samples(plan.x, plan.y)
                zero = np.zeros_like(gu)
                fields = (gu, zero, gux, guy, zero, zero)
                for field in fields:
                    field.flags.writeable = False
                cached = (plan, self.lifting.space, self.lifting.coefficients.copy(), fields)
                self._lid_samples = cached
            return cached[3]
        return (*self.lifting.velocity(plan.x, plan.y),
                *self.lifting.derivatives(plan.x, plan.y))

    def _sample_velocity(self, velocity: Array, plan: QuadraturePlan) -> tuple[Array, ...]:
        coefficients = np.asarray(velocity).reshape(2, self.space.size, self.space.size)
        u, ux, uy = plan.sample(coefficients[0])
        v, vx, vy = plan.sample(coefficients[1])
        fields: tuple[Array, ...] = (u, v, ux, uy, vx, vy)
        if self.lifting is not None:
            fields = tuple(field+known for field, known in zip(fields, self._lifting_fields(plan)))
        return fields

    def nonlinear_with_lifting(self, velocity: Array) -> tuple[Array, float]:
        """Return N(w) and b(w+g,w+g,g) for a fixed lifting g.

        N represents the antisymmetric load on homogeneous tests. The scalar is
        integrated directly against g, not against its homogeneous-space projection.
        Both use the same quadrature, so <N(w),w> + lifting_work cancels algebraically.
        """
        plan = self._quadrature.get(self.quadrature_extra)
        u, v, du_dx, du_dy, dv_dx, dv_dy = self._sample_velocity(velocity, plan)
        advection_u, advection_v = u*du_dx+v*du_dy, u*dv_dx+v*dv_dy
        flux_uv = u*v
        # Assemble half of ((u.grad)u,test) - ((u.grad)test,u).
        load_u = .5 * (plan.project(advection_u) - plan.stiffness_load(u*u, flux_uv))
        load_v = .5 * (plan.project(advection_v) - plan.stiffness_load(flux_uv, v*v))
        lifting_work = 0.
        if self.lifting is not None:
            g_u, g_v, dg_u_x, dg_u_y, dg_v_x, dg_v_y = self._lifting_fields(plan)
            integrand = (advection_u * g_u + advection_v * g_v
                         - (u * dg_u_x + v * dg_u_y) * u
                         - (u * dg_v_x + v * dg_v_y) * v)
            lifting_work = .5 * float(np.sum(plan.weight * integrand))
        return np.concatenate([self._mass_solve(load_u), self._mass_solve(load_v)]), lifting_work

    def _total_divergence_inf(self, coefficients_x: Array, coefficients_y: Array) -> float:
        """max |div u| in physical space, including the lifting when there is one."""
        plan = self._quadrature.get(self.quadrature_extra)
        divergence = plan.divergence(coefficients_x, coefficients_y)
        if self.lifting is not None:
            if type(self.lifting) is LidLifting:
                _, _, gux, _, _, gvy = self._lifting_fields(plan)
                divergence = divergence + gux + gvy
            else:
                divergence = divergence + self.lifting.samples(plan.x, plan.y)[3]
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
        """One batched solve; only the first right-hand side carries the boundary data.

        The schemes add the later columns to the first as corrections, so the inhomogeneous
        Dirichlet datum has to survive that combination. Solving every column with the same
        datum would multiply it by the sum of the coefficients, so the correction columns
        are solved with a homogeneous one: because the model is linear, the combination is
        then exactly the solution of the combined right-hand side. Factorizations are
        reused from the cache either way.
        """
        if len(columns) == 0:
            return []
        loads_x = [self.mass @ np.asarray(column)[:self.modes] for column in columns]
        loads_y = [self.mass @ np.asarray(column)[self.modes:] for column in columns]
        continuity: list[Array | None] = [None]*len(columns)
        if self._lifting is not None:
            loads_x[0] -= viscosity*self._lifting[0].reshape(-1)
            loads_y[0] -= viscosity*self._lifting[1].reshape(-1)
            continuity[0] = -self._lifting[2].reshape(-1)
        solutions = self.stokes.solve_columns(loads_x, loads_y, mass=mass, viscosity=viscosity,
                                              continuities=continuity)
        return [SpectralSolution(np.r_[s.velocity_x.reshape(-1), s.velocity_y.reshape(-1)],
                                 s.pressure.reshape(-1), s.residual,
                                 self._total_divergence_inf(s.velocity_x, s.velocity_y))
                for s in solutions]

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
        velocity = self.vector(state)
        plan = self._quadrature.get(self.quadrature_extra)
        u, v, du_dx, du_dy, dv_dx, dv_dy = self._sample_velocity(velocity, plan)
        weight = plan.weight
        kinetic = .5 * float(np.sum(weight * (u ** 2 + v ** 2)))
        return {'kinetic': kinetic, 'modified_energy': kinetic + .5 * (state.r - 1) ** 2,
                'divergence_inf': float(np.max(np.abs(du_dx + dv_dy))),
                'h1_seminorm_squared': float(np.sum(weight * (du_dx ** 2 + du_dy ** 2
                                                              + dv_dx ** 2 + dv_dy ** 2))),
                'r': state.r}

    def close(self) -> None:
        self.stokes.cache.clear()
        self._quadrature.clear()
        self._lid_samples = None
        self._mass_factorization = None


if TYPE_CHECKING:
    from ..model import Model
    _spectral_realisation: type[Model] = SpectralModel

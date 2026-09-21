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
                 tolerance: float = 1e-10, cache_size: int = 4, dealias: int = 8) -> None:
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
        self.force: Callable[[float], Array] = force if force is not None else lambda t: self.zero_velocity()

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
        """The weak Laplacian as a field operator: M^-1 S, not S.

        S is the weak stiffness, i.e. it maps coefficients to a *load*. The seam adds the
        result of this call to other fields and hands the sum to `solve`, so it has to
        return a field, and the mass matrix has to be folded in. The MAC model never has to
        make this choice because its basis functions are nodal indicators and its mass
        matrix is diagonal; a modal basis does, and picking the wrong convention is a
        silent O(1) error rather than a visible one.
        """
        modes = self.modes
        out = np.empty_like(np.asarray(velocity))
        out[:modes] = self._mass_solve(self.stiffness @ velocity[:modes])
        out[modes:] = self._mass_solve(self.stiffness @ velocity[modes:])
        return out

    def apply_G(self, pressure: Array) -> Array:
        """The gradient in the field representation, M^-1 G with G = -D^T, per component."""
        return np.concatenate([self._mass_solve(-(self.divergence_x.T @ pressure)),
                               self._mass_solve(-(self.divergence_y.T @ pressure))])

    def apply_D(self, velocity: Array) -> Array:
        modes = self.modes
        return self.divergence_x @ velocity[:modes] + self.divergence_y @ velocity[modes:]

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
        """N(v) = (u.grad)u, assembled on the wide quadrature, projected and solved."""
        space, modes = self.space, self.modes
        size = space.size
        nodes_x, _, nodes_y, _ = space.nodes(extra=self.dealias)
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
        return np.concatenate([self._mass_solve(space.project(u * du_dx + v * du_dy, extra=self.dealias)),
                               self._mass_solve(space.project(u * dv_dx + v * dv_dy, extra=self.dealias))])

    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> SolveResult:
        """Solve the shifted Stokes problem for a forcing given in the field representation.

        The system is [nu S, G; D, 0] [c; p] = [load; 0] with S, G, D the weak matrices, so
        the incoming field is turned into a load by the mass matrix before the solve and
        the coefficients that come back need no further conversion.
        """
        modes = self.modes
        size = self.space.size
        load = self.mass @ np.asarray(rhs)[:modes]
        solution = self.stokes.solve(load.reshape(size, size), (self.mass @ np.asarray(rhs)[modes:]).reshape(size, size),
                                     mass=mass, viscosity=viscosity)
        return SpectralSolution(np.concatenate([solution.velocity_x.reshape(-1), solution.velocity_y.reshape(-1)]),
                                solution.pressure.reshape(-1), solution.residual, solution.divergence_inf)

    def solve_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float) -> list[SolveResult]:
        """One solve per right-hand side; the factorization is reused from the cache."""
        return [self.solve(column, mass=mass, viscosity=viscosity) for column in columns]

    def stage(self, pressure: Array, residual: float, divergence_inf: float, r: float = 0.,
              candidates: Sequence[float] = (), root_residuals: Sequence[float] = (),
              scalar_residual: float = 0.) -> Stage:
        return Stage(np.asarray(pressure), residual, divergence_inf, r, list(candidates),
                     list(root_residuals), scalar_residual)

    def diagnostics(self, state: State) -> dict[str, float]:
        velocity = self.vector(state)
        kinetic = .5 * self.inner(velocity, velocity)
        return {'kinetic': kinetic, 'modified_energy': kinetic + .5 * (state.r - 1) ** 2,
                'divergence_inf': self.stokes.divergence_inf(np.asarray(state.u), np.asarray(state.v)),
                'h1_seminorm_squared': self.inner(velocity, self.apply_K(velocity)), 'r': state.r}

    def close(self) -> None:
        self.stokes.cache.clear()


if TYPE_CHECKING:
    from ..model import Model
    _spectral_realisation: type[Model] = SpectralModel

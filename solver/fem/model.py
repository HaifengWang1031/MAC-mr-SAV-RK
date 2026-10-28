"""P2/P1 adapter: free-DOF velocity coefficients, full component P2 State.u/v."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any
import numpy as np
from scipy.sparse.linalg import splu
from skfem import BilinearForm, LinearForm, asm
from skfem.helpers import ddot, div, dot, grad
from ..core import Stage, State
from ..model import SolveResult
from .space import Array, FiniteElementSpace
from .stokes import FEMStokes
from .quadrature import P2Quadrature

VectorField = Callable[[Array, Array], tuple[Array | float, Array | float]]


@BilinearForm
def _mass(u: Any, v: Any, w: Any) -> Any:
    return dot(u, v)


@BilinearForm
def _stiffness(u: Any, v: Any, w: Any) -> Any:
    return ddot(grad(u), grad(v))


@BilinearForm
def _divergence(u: Any, q: Any, w: Any) -> Any:
    return div(u)*q


@LinearForm
def _integral(q: Any, w: Any) -> Any:
    return q


@dataclass
class FEMSolution:
    velocity: Array
    pressure: Array
    residual: float
    divergence_inf: float
    continuity_residual: float


class FEMModel:
    def __init__(self, space: FiniteElementSpace, nu: float, *, force: Callable[[float], Array] | None = None,
                 tolerance: float = 1e-9, cache_size: int = 4,
                 dirichlet_facets: np.ndarray | None = None, pressure_gauge: bool = True,
                 stokes_backend: str = 'direct') -> None:
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError('nu must be positive and finite')
        self.space = space
        self.nu = nu
        self.velocity_basis, self.pressure_basis = space.bases()
        if dirichlet_facets is not None:
            facets = np.asarray(dirichlet_facets)
            if facets.ndim != 1 or facets.dtype.kind not in 'iu' or not facets.size \
                    or not np.isin(facets, self.velocity_basis.mesh.boundary_facets()).all():
                raise ValueError('Dirichlet facets must be a nonempty boundary subset')
        self.boundary_dofs = self.velocity_basis.get_dofs(facets=dirichlet_facets).all()
        self.free_dofs = self.velocity_basis.complement_dofs(self.boundary_dofs)
        self.component_dofs = tuple(self.velocity_basis.split_indices())
        self.pressure_gauge = pressure_gauge
        self._boundary_values = np.zeros(self.velocity_basis.N)
        if len(self.free_dofs) < self.pressure_basis.N-int(pressure_gauge):
            raise ValueError('Mesh has too few free velocity DOFs for the pressure constraints')
        self.full_mass = asm(_mass, self.velocity_basis).tocsc()
        self.full_stiffness = asm(_stiffness, self.velocity_basis).tocsc()
        self.full_divergence = asm(_divergence, self.velocity_basis, self.pressure_basis).tocsr()
        self.mass = self.full_mass[self.free_dofs][:, self.free_dofs].tocsc()
        self.stiffness = self.full_stiffness[self.free_dofs][:, self.free_dofs].tocsc()
        self.divergence = self.full_divergence[:, self.free_dofs].tocsr()
        self.pressure_weights = asm(_integral, self.pressure_basis)
        self._quadrature = P2Quadrature(self.velocity_basis)
        self._mass_factor = splu(self.mass, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.)
        self._nonlinear_calls, self._projection_calls = 0, 0
        self._nonlinear_wall_seconds, self._projection_wall_seconds = 0., 0.
        self.stokes = FEMStokes(self.mass, self.stiffness, self.divergence, self.pressure_weights,
                                tolerance=tolerance, cache_size=cache_size, pressure_gauge=pressure_gauge,
                                backend=stokes_backend, mass_solve=self._mass_factor.solve)
        size = len(self.free_dofs)
        self.force: Callable[[float], Array] = force if force is not None else lambda t: np.zeros(size)

    def performance_metrics(self) -> dict[str, float | int]:
        return {**{'stokes_'+key: value for key, value in self.stokes.metrics().items()},
                **{'volume_'+key: value for key, value in self._quadrature.metrics().items()},
                'nonlinear_calls': self._nonlinear_calls, 'nonlinear_wall_seconds': self._nonlinear_wall_seconds,
                'projection_calls': self._projection_calls, 'projection_wall_seconds': self._projection_wall_seconds}

    def zero_velocity(self) -> Array:
        return np.zeros(len(self.free_dofs))

    def zero_pressure(self) -> Array:
        return np.zeros(self.pressure_basis.N)

    def _velocity(self, velocity: Array) -> Array:
        coefficients = np.asarray(velocity, dtype=float)
        if coefficients.shape != (len(self.free_dofs),) or not np.isfinite(coefficients).all():
            raise ValueError('Invalid free velocity coefficients')
        return coefficients

    def _pressure(self, pressure: Array) -> Array:
        coefficients = np.asarray(pressure, dtype=float)
        if coefficients.shape != (self.pressure_basis.N,) or not np.isfinite(coefficients).all():
            raise ValueError('Invalid pressure coefficients')
        return coefficients

    def full_velocity(self, velocity: Array) -> Array:
        """Expand free coefficients, including the model's known Dirichlet trace."""
        full = self._boundary_values.copy()
        full[self.free_dofs] = self._velocity(velocity)
        return full

    def project(self, function: VectorField) -> Array:
        """Constrained consistent-L2 projection (not interpolation or Leray projection)."""
        start = perf_counter()
        self._projection_calls += 1
        try:
            x = self.velocity_basis.global_coordinates()
            components = function(x[0], x[1])
            if len(components) != 2:
                raise ValueError('Field must have two components')
            values = np.stack([np.broadcast_to(np.asarray(c, dtype=float), x[0].shape) for c in components])
            if not np.isfinite(values).all():
                raise ValueError('Field must return finite broadcastable components')
            return self._velocity(self._mass_factor.solve(self._quadrature.load(values)[self.free_dofs]))
        finally:
            self._projection_wall_seconds += perf_counter()-start

    def project_pressure(self, function: Callable[[Array, Array], Array | float]) -> Array:
        """Consistent P1 projection; integral-zero only for a closed pressure gauge."""
        @BilinearForm
        def mass(p: Any, q: Any, w: Any) -> Any:
            return p*q

        @LinearForm
        def load(q: Any, w: Any) -> Any:
            values = np.broadcast_to(np.asarray(function(w.x[0], w.x[1]), dtype=float), w.x[0].shape)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite pressure field')
            return values*q

        coefficients = splu(asm(mass, self.pressure_basis).tocsc()).solve(asm(load, self.pressure_basis))
        if self.pressure_gauge:
            coefficients -= self.pressure_integral(coefficients)/np.sum(self.pressure_weights)
        return self._pressure(coefficients)

    def _convection_load(self, full: Array) -> Array:
        return self._quadrature.convection_load(full)

    def nonlinear(self, velocity: Array) -> Array:
        start = perf_counter()
        self._nonlinear_calls += 1
        try:
            load = self._convection_load(self.full_velocity(velocity))[self.free_dofs]
            coefficients = self._mass_factor.solve(load)
            if not np.isfinite(coefficients).all():
                raise FloatingPointError('Nonfinite FEM convection')
            return coefficients
        finally:
            self._nonlinear_wall_seconds += perf_counter()-start

    def nonlinear_with_lifting(self, velocity: Array) -> tuple[Array, float]:
        return self.nonlinear(velocity), 0.

    def vector(self, state: State) -> Array:
        if not np.isfinite([state.t, state.r]).all():
            raise ValueError('State time and r must be finite')
        full = np.zeros(self.velocity_basis.N)
        for values, indices in zip((state.u, state.v), self.component_dofs):
            if np.shape(values) != indices.shape or not np.isfinite(values).all():
                raise ValueError('Invalid full component coefficients')
            full[indices] = values
        if np.any(full[self.boundary_dofs] != self._boundary_values[self.boundary_dofs]):
            raise ValueError('Unexpected boundary velocity coefficients')
        return full[self.free_dofs].copy()

    def state(self, t: float, velocity: Array, r: float = 0.) -> State:
        if not np.isfinite([t, r]).all():
            raise ValueError('State time and r must be finite')
        full = self.full_velocity(velocity)
        return State(t, full[self.component_dofs[0]].copy(), full[self.component_dofs[1]].copy(), r)

    def _points(self, points: Array) -> Array:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[0] != 2 or not points.shape[1] or not np.isfinite(points).all():
            raise ValueError('Expected finite physical points with shape (2, n), n > 0')
        return points

    def evaluate_velocity(self, velocity: Array, points: Array) -> Array:
        return np.asarray(self.velocity_basis.interpolator(self.full_velocity(velocity))(self._points(points)))

    def evaluate_gradient(self, velocity: Array, points: Array) -> Array:
        """Physical d_j u_i, shape (component, derivative, point).

        At element interfaces the mesh finder selects one adjacent trace.
        """
        points = self._points(points)
        basis = self.velocity_basis
        coefficients = self.full_velocity(velocity)
        cells = basis.mesh.element_finder(mapping=basis.mapping)(*points)
        reference = basis.mapping.invF(points[:, :, None], tind=cells)
        gradient = np.zeros((2, 2, points.shape[1]))
        for k in range(basis.Nbfun):
            field = basis.elem.gbasis(basis.mapping, reference, k, tind=cells)[0]
            gradient += field.grad[:, :, :, 0]*coefficients[basis.element_dofs[k, cells]]
        return gradient

    def evaluate_pressure(self, pressure: Array, points: Array) -> Array:
        return np.asarray(self.pressure_basis.interpolator(self._pressure(pressure))(self._points(points)))

    def pressure_integral(self, pressure: Array) -> float:
        return float(self.pressure_weights @ self._pressure(pressure))

    def apply_D(self, velocity: Array) -> Array:
        return np.asarray(self.divergence @ self._velocity(velocity))

    def max_abs(self, field: Array) -> float:
        if not np.isfinite(field).all():
            raise ValueError('Nonfinite field')
        return float(np.max(np.abs(field)))

    def apply_K(self, velocity: Array) -> Array:
        return self._mass_factor.solve(self.stiffness @ self._velocity(velocity))

    def apply_G(self, pressure: Array) -> Array:
        return self._mass_factor.solve(-self.divergence.T @ self._pressure(pressure))

    def inner(self, left: Array, right: Array) -> float:
        return float(self._velocity(left) @ (self.mass @ self._velocity(right)))

    def combine(self, *terms: tuple[float, Array]) -> Array:
        if not terms:
            raise ValueError('At least one term is required')
        total = np.zeros_like(terms[0][1], dtype=float)
        for coefficient, field in terms:
            if np.shape(field) != total.shape or not np.isfinite(coefficient) or not np.isfinite(field).all():
                raise ValueError('Invalid linear combination')
            total += coefficient*field
        if not np.isfinite(total).all():
            raise FloatingPointError('Nonfinite linear combination')
        return total

    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> SolveResult:
        return self.solve_columns([rhs], mass=mass, viscosity=viscosity)[0]

    def solve_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float) -> list[SolveResult]:
        results = self.stokes.solve_columns([self._velocity(c) for c in columns], mass=mass, viscosity=viscosity)
        return [FEMSolution(result.velocity, result.pressure, result.residual,
                            self.physical_divergence_inf(result.velocity), result.continuity_residual)
                for result in results]

    def stage_divergences(self, velocity: Array) -> tuple[float, float]:
        return self.physical_divergence_inf(velocity), self.max_abs(self.apply_D(velocity))

    def physical_divergence_inf(self, velocity: Array) -> float:
        """Maximum at assembly quadrature points, not an analytic supremum."""
        _, gradient = self._quadrature.evaluate(self.full_velocity(velocity))
        return float(np.max(np.abs(gradient[0, 0]+gradient[1, 1])))

    def stage(self, pressure: Array, residual: float, divergence_inf: float, r: float = 0.,
              candidates: Sequence[float] = (), root_residuals: Sequence[float] = (),
              scalar_residual: float = 0., *, continuity_residual: float = float('nan')) -> Stage:
        return Stage(self._pressure(pressure).copy(), residual, divergence_inf, r, list(candidates),
                     list(root_residuals), scalar_residual, continuity_residual)

    def diagnostics(self, state: State) -> dict[str, float]:
        velocity = self.vector(state)
        _, gradient = self._quadrature.evaluate(self.full_velocity(velocity))
        divergence = gradient[0, 0]+gradient[1, 1]
        weight = self.velocity_basis.dx
        kinetic = .5*self.inner(velocity, velocity)
        return {'kinetic': kinetic, 'modified_energy': kinetic+.5*(state.r-1)**2,
                'divergence_inf': float(np.max(np.abs(divergence))),
                'divergence_l2': float(np.sqrt(np.sum(divergence**2*weight))),
                'continuity_residual': self.max_abs(self.apply_D(velocity)),
                'h1_seminorm_squared': float(velocity @ (self.stiffness @ velocity)), 'r': state.r}

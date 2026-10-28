"""Fixed Dirichlet inflow and gradient-traction outlet behind the existing Model."""
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

import numpy as np
from skfem import FacetBasis

from ..core import State
from ..model import SolveResult
from .model import FEMModel, FEMSolution, VectorField
from .space import Array, FiniteElementSpace
from .stokes import FEMStokes
from .quadrature import P2Quadrature


class OpenFlowModel(FEMModel):
    """Physical free coefficients plus a fixed known boundary trace.

    Natural outlet pressure is unique, not mean-zero. The first solve column
    carries the trace; SAV correction columns carry zero trace/continuity RHS.
    """
    def __init__(self, space: FiniteElementSpace, nu: float, *, dirichlet_facets: np.ndarray,
                 boundary_velocity: VectorField, force: Callable[[float], Array] | None = None,
                 tolerance: float = 1e-9, cache_size: int = 2, stokes_backend: str = 'direct') -> None:
        super().__init__(space, nu, force=force, tolerance=tolerance, cache_size=cache_size,
                         dirichlet_facets=dirichlet_facets, pressure_gauge=False, stokes_backend=stokes_backend)
        if np.isin(self.velocity_basis.mesh.boundary_facets(), dirichlet_facets).all():
            raise ValueError('Natural outlet requires unconstrained boundary facets')
        x, y = self.velocity_basis.doflocs
        values = boundary_velocity(x, y)
        if len(values) != 2:
            raise ValueError('Boundary velocity must have two components')
        for component, indices in zip(values, self.component_dofs):
            values_at_nodes = np.broadcast_to(np.asarray(component, dtype=float), x.shape)
            if not np.isfinite(values_at_nodes).all():
                raise ValueError('Nonfinite boundary velocity')
            self._boundary_values[indices] = values_at_nodes[indices]
        self._boundary_values[self.free_dofs] = 0.
        self._boundary_values.flags.writeable = False
        self._boundary_K = self._mass_factor.solve(
            self.full_stiffness[self.free_dofs] @ self._boundary_values)
        self._boundary_D = np.asarray(self.full_divergence @ self._boundary_values)
        mesh = self.velocity_basis.mesh
        self._facet_order = max(6, space.quadrature_order)
        self._all_facets = FacetBasis(mesh, self.velocity_basis.elem, facets=mesh.boundary_facets(),
                                     intorder=self._facet_order)
        self._boundary_quadrature = P2Quadrature(self._all_facets)
        self._named_facets: dict[str, tuple[Any, Any, np.ndarray]] = {}
        self._pressure_stokes = FEMStokes(self.mass, self.stiffness, self.divergence,
                                          self.pressure_weights, tolerance=tolerance,
                                          cache_size=1, pressure_gauge=False, mass_solve=self._mass_factor.solve)

    def performance_metrics(self) -> dict[str, float | int]:
        return {**super().performance_metrics(),
                **{'boundary_'+key: value for key, value in self._boundary_quadrature.metrics().items()},
                **{'pressure_'+key: value for key, value in self._pressure_stokes.metrics().items()}}

    def apply_K(self, velocity: Array) -> Array:
        return super().apply_K(velocity)+self._boundary_K

    def apply_D(self, velocity: Array) -> Array:
        return super().apply_D(velocity)+self._boundary_D

    def nonlinear_with_lifting(self, velocity: Array) -> tuple[Array, float]:
        start = perf_counter()
        self._nonlinear_calls += 1
        try:
            full = self.full_velocity(velocity)
            load = self._convection_load(full)
            field, _ = self._boundary_quadrature.evaluate(full)
            normal_velocity = np.sum(field*self._all_facets.normals, axis=0)
            load += self._boundary_quadrature.load(.5*normal_velocity[None]*field)
            boundary_power = .5*float(np.sum(normal_velocity*np.sum(field**2, axis=0)*self._all_facets.dx))
            # Known-trace work minus physical convective energy crossing the boundary.
            lifting = float(load @ self._boundary_values)-boundary_power
            coefficients = self._mass_factor.solve(load[self.free_dofs])
            if not np.isfinite(coefficients).all() or not np.isfinite(lifting):
                raise FloatingPointError('Nonfinite open-flow convection')
            return coefficients, lifting
        finally:
            self._nonlinear_wall_seconds += perf_counter()-start

    def nonlinear(self, velocity: Array) -> Array:
        return self.nonlinear_with_lifting(velocity)[0]

    def solve_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float) -> list[SolveResult]:
        if not len(columns):
            return super().solve_columns(columns, mass=mass, viscosity=viscosity)
        adjusted = [self._velocity(c) for c in columns]
        adjusted[0] = adjusted[0]-viscosity*self._boundary_K
        constraint = np.zeros((self.pressure_basis.N, len(columns)))
        constraint[:, 0] = -self._boundary_D
        solutions = self.stokes.solve_columns(adjusted, mass=mass, viscosity=viscosity,
                                              divergence_rhs=constraint)
        results: list[SolveResult] = []
        for index, solution in enumerate(solutions):
            full = self.full_velocity(solution.velocity) if index == 0 else np.zeros(self.velocity_basis.N)
            if index:
                full[self.free_dofs] = solution.velocity
            _, gradient = self._quadrature.evaluate(full)
            divergence = float(np.max(np.abs(gradient[0, 0]+gradient[1, 1])))
            results.append(FEMSolution(solution.velocity, solution.pressure, solution.residual,
                                       divergence, solution.continuity_residual))
        return results

    def instantaneous_solution(self, velocity: Array, t: float) -> tuple[Array, Array]:
        """Constrained velocity rate and physical semidiscrete pressure, not a stage increment."""
        if not np.isfinite(t):
            raise ValueError('Nonfinite observation time')
        rhs = self.combine((1., self.force(t)), (-self.nu, self.apply_K(velocity)),
                           (-1., self.nonlinear(velocity)))
        # The fixed trace has zero time derivative, hence a homogeneous rate solve.
        solved = self._pressure_stokes.solve_columns([rhs], mass=1., viscosity=0.)[0]
        return solved.velocity, solved.pressure

    def _facet_bases(self, boundary: str) -> tuple[Any, Any, np.ndarray]:
        if boundary not in self._named_facets:
            mesh = self.velocity_basis.mesh
            if mesh.boundaries is None or boundary not in mesh.boundaries:
                raise ValueError(f'Unknown named boundary: {boundary}')
            facets = mesh.boundaries[boundary]
            vb = FacetBasis(mesh, self.velocity_basis.elem, facets=facets, intorder=self._facet_order)
            positions = np.searchsorted(self._all_facets.find, vb.find)
            if np.any(positions >= len(self._all_facets.find)) \
                    or not np.array_equal(self._all_facets.find[positions], vb.find):
                raise ValueError('Force boundary must be exterior')
            self._named_facets[boundary] = (vb,
                FacetBasis(mesh, self.pressure_basis.elem, facets=facets, intorder=self._facet_order), positions)
        return self._named_facets[boundary]

    def boundary_force(self, velocity: Array, pressure: Array, boundary: str) -> Array:
        """Fluid force on the solid: (nu grad(u)-p I) times the solid outward normal."""
        vb, pb, positions = self._facet_bases(boundary)
        _, gradient = self._boundary_quadrature.evaluate(self.full_velocity(velocity))
        p = pb.interpolate(self._pressure(pressure))
        derivative = np.einsum('ijef,jef->ief', gradient[:, :, positions], vb.normals)
        return np.sum((p[None]*vb.normals-self.nu*derivative)*vb.dx[None], axis=(1, 2))

    def diagnostics(self, state: State) -> dict[str, float]:
        velocity = self.vector(state)
        full = self.full_velocity(velocity)
        _, gradient = self._quadrature.evaluate(full)
        divergence = gradient[0, 0]+gradient[1, 1]
        boundary, _ = self._boundary_quadrature.evaluate(full)
        normal_velocity = np.sum(boundary*self._all_facets.normals, axis=0)
        kinetic = .5*float(full @ (self.full_mass @ full))
        return {'kinetic': kinetic, 'modified_energy': kinetic+.5*(state.r-1)**2,
                'h1_seminorm_squared': float(full @ (self.full_stiffness @ full)),
                'divergence_inf': float(np.max(np.abs(divergence))),
                'divergence_l2': float(np.sqrt(np.sum(divergence**2*self.velocity_basis.dx))),
                'continuity_residual': self.max_abs(self.apply_D(velocity)), 'r': state.r,
                'boundary_flux': float(np.sum(normal_velocity*self._all_facets.dx)),
                'convective_boundary_power': .5*float(np.sum(
                    normal_velocity*np.sum(boundary**2, axis=0)*self._all_facets.dx))}

"""Closed-cavity driving behind the existing Model seam.

`OpenFlowModel` requires at least one unconstrained boundary facet because the
natural-outlet pressure is then unique without a gauge. A lid-driven cavity has
the opposite arrangement: every facet carries a Dirichlet velocity, the net flux
is zero, and the pressure is only defined up to a constant, so the model keeps
`pressure_gauge=True` and drops the natural-outlet treatment entirely.

The only non-homogeneous datum is the sliding lid. As in `OpenFlowModel` it is
carried as a fixed trace, folded into two cached vectors:

    _boundary_K = M^-1 (K_fb u_b)      _boundary_D = D_fb u_b

`apply_K`/`apply_D` therefore act on the full field, while `solve_columns`
subtracts the trace contribution from the first right-hand side only, because
the second (SAV response) column solves with a homogeneous trace and homogeneous
continuity. Everything else — the SAV algebra, the adaptivity controller and the
schemes — sees an ordinary model.
"""
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from skfem import FacetBasis

from ..core import State
from ..model import SolveResult
from .model import FEMModel, FEMSolution
from .space import Array, FiniteElementSpace
from .quadrature import P2Quadrature

LID = 'lid'


class ClosedCavityModel(FEMModel):
    """All-Dirichlet closed domain with one sliding wall and a mean-zero pressure."""

    def __init__(self, space: FiniteElementSpace, nu: float, *, lid_speed: float = 1.,
                 lid_boundary: str = LID, force: Callable[[float], Array] | None = None,
                 tolerance: float = 1e-9, cache_size: int = 2, stokes_backend: str = 'direct',
                 stokes_ordering: str = 'auto', stokes_factor_backend: str = 'superlu',
                 stokes_threads: int | None = None, corner_policy: str = 'stationary',
                 mass_backend: str = 'direct', stokes_amg_type: str = 'gamg', fem_load_mode: str = 'riesz') -> None:
        if not np.isfinite(lid_speed) or lid_speed <= 0.:
            raise ValueError('lid_speed must be finite and positive')
        if corner_policy not in ('stationary', 'lid'):
            raise ValueError('corner_policy must be stationary or lid')
        super().__init__(space, nu, force=force, tolerance=tolerance, cache_size=cache_size,
                         pressure_gauge=True,
                         stokes_backend=stokes_backend, stokes_ordering=stokes_ordering,
                         stokes_factor_backend=stokes_factor_backend, stokes_threads=stokes_threads,
                         mass_backend=mass_backend, stokes_amg_type=stokes_amg_type, fem_load_mode=fem_load_mode)
        mesh = self.velocity_basis.mesh
        if mesh.boundaries is None or lid_boundary not in mesh.boundaries:
            raise ValueError(f'Unknown named lid boundary: {lid_boundary}')
        self.lid_speed = float(lid_speed)
        self.lid_boundary = lid_boundary
        self.corner_policy = corner_policy
        self._set_lid_trace(mesh)
        self._boundary_K = self._mass_factor.solve(
            self.full_stiffness[self.free_dofs] @ self._boundary_values)
        self._boundary_D = np.asarray(self.full_divergence @ self._boundary_values)
        self._facet_order = max(6, space.quadrature_order)
        self._all_facets = FacetBasis(mesh, self.velocity_basis.elem, facets=mesh.boundary_facets(),
                                     intorder=self._facet_order)
        self._boundary_quadrature = P2Quadrature(self._all_facets)
        self._named_facets: dict[str, tuple[Any, Any]] = {}

    def _set_lid_trace(self, mesh: Any) -> None:
        """Tangential unit velocity on the lid facets, zero everywhere else.

        Only the tangential component is set: a lid that slides along y = const has
        zero normal velocity, so the closed domain stays compatible (net flux zero).
        """
        ends = mesh.p[:, mesh.facets[:, mesh.boundaries[self.lid_boundary]]]
        if np.ptp(ends[1]) > 1e-12:
            raise ValueError('Lid facets must be horizontal to carry a purely tangential trace')
        x, y = self.velocity_basis.doflocs
        on_lid = np.isclose(y, float(ends[1].mean()), rtol=0., atol=1e-12)
        trace = np.zeros(self.velocity_basis.N)
        for component, indices in enumerate(self.component_dofs):
            values = np.zeros_like(x)
            if component == 0:
                values[on_lid] = self.lid_speed
            trace[indices] = values[indices]
        if self.corner_policy == 'stationary':
            # Shared end vertices belong to stationary walls. This keeps the
            # complete P2 trace zero along both inclined walls, not only midpoints.
            walls = np.setdiff1d(mesh.boundary_facets(), mesh.boundaries[self.lid_boundary])
            trace[self.velocity_basis.get_dofs(facets=walls).all()] = 0.
        trace[self.free_dofs] = 0.
        if not np.any(trace):
            raise ValueError('Empty lid trace')
        self._boundary_values = trace
        self._boundary_values.flags.writeable = False

    def performance_metrics(self) -> dict[str, float | int]:
        return {**super().performance_metrics(),
                **{'boundary_'+key: value for key, value in self._boundary_quadrature.metrics().items()}}

    def apply_K(self, velocity: Array) -> Array:
        return super().apply_K(velocity)+self._boundary_K

    def apply_D(self, velocity: Array) -> Array:
        return super().apply_D(velocity)+self._boundary_D

    def _convection_load_with_lifting(self, velocity: Array) -> tuple[Array, float]:
        # Tangential driving removes boundary flux, but not the volume pairing
        # b(w+g,w+g,g). Use the same skew quadrature as N(w), so <N(w),w>+b(U,U,g)=0.
        load = self._convection_load(self.full_velocity(velocity))
        return load[self.free_dofs], float(load @ self._boundary_values)

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
            if index == 0:
                divergence = float(np.max(np.abs(self._quadrature.divergence(
                    self.full_velocity(solution.velocity), cache_field=len(solutions) == 1))))
            else:
                full = np.zeros(self.velocity_basis.N)
                full[self.free_dofs] = solution.velocity
                divergence = float(np.max(np.abs(self._quadrature.divergence(full))))
            results.append(FEMSolution(solution.velocity, solution.pressure, solution.residual,
                                       divergence, solution.continuity_residual))
        return results

    def _facet_bases(self, boundary: str) -> tuple[Any, Any]:
        if boundary not in self._named_facets:
            mesh = self.velocity_basis.mesh
            if mesh.boundaries is None or boundary not in mesh.boundaries:
                raise ValueError(f'Unknown named boundary: {boundary}')
            facets = mesh.boundaries[boundary]
            self._named_facets[boundary] = (
                FacetBasis(mesh, self.velocity_basis.elem, facets=facets, intorder=self._facet_order),
                FacetBasis(mesh, self.pressure_basis.elem, facets=facets, intorder=self._facet_order))
        return self._named_facets[boundary]

    def boundary_force(self, velocity: Array, pressure: Array, boundary: str) -> Array:
        """Fluid force on the solid: (nu grad(u) - p I) n, with n the solid's outward normal."""
        vb, pb = self._facet_bases(boundary)
        _, gradient = self._boundary_quadrature.evaluate(self.full_velocity(velocity))
        positions = np.searchsorted(self._all_facets.find, vb.find)
        if np.any(positions >= len(self._all_facets.find)) \
                or not np.array_equal(self._all_facets.find[positions], vb.find):
            raise ValueError('Force boundary must be exterior')
        p = pb.interpolate(self._pressure(pressure))
        derivative = np.einsum('ijef,jef->ief', gradient[:, :, positions], vb.normals)
        return np.sum((p[None]*vb.normals-self.nu*derivative)*vb.dx[None], axis=(1, 2))

    def diagnostics(self, state: State) -> dict[str, float]:
        velocity = self.vector(state)
        full = self.full_velocity(velocity)
        _, gradient = self._quadrature.evaluate(full)
        divergence = gradient[0, 0]+gradient[1, 1]
        kinetic = .5*float(full @ (self.full_mass @ full))
        return {'kinetic': kinetic, 'modified_energy': kinetic+.5*(state.r-1)**2,
                'h1_seminorm_squared': float(full @ (self.full_stiffness @ full)),
                'divergence_inf': float(np.max(np.abs(divergence))),
                'divergence_l2': float(np.sqrt(np.sum(divergence**2*self.velocity_basis.dx))),
                'continuity_residual': self.max_abs(self.apply_D(velocity)), 'r': state.r}

"""Pressure-driven MWR thin-gap model behind the existing time-stepping interface.

Physical coefficients (a_t,a_c,a_v,d)=(6/5,54/35,6/5,12). Divide momentum by
a_t*rho: nu_eff=nu, beta=9/7, drag=10*nu/gap**2, pi=p/(a_t*rho).
Reference: https://arxiv.org/html/2511.03956v2, equations 18--19a.
This is a single-profile reduced model, not a full three-dimensional simulation.
"""
from collections.abc import Callable
from typing import Any

import numpy as np
from skfem import FacetBasis, LinearForm, asm

from ..core import State
from .open_flow import OpenFlowModel
from .space import Array, FiniteElementSpace

TIME_COEFFICIENT = 6./5.
CONVECTION_COEFFICIENT = 9./7.


class HeleShawModel(OpenFlowModel):
    """Zero-trace walls/foil and prescribed pressure tractions at both open ends."""

    def __init__(self, space: FiniteElementSpace, nu: float, gap: float, *,
                 pressure_drop: float = 1., ramp_time: float = 0., rho: float = 1.,
                 convection: bool = True, outlet_treatment: str = 'natural',
                 inlet_treatment: str = 'natural',
                 tolerance: float = 1e-9, cache_size: int = 2,
                 stokes_backend: str = 'direct', stokes_ordering: str = 'auto',
                 mass_backend: str = 'direct', stokes_amg_type: str = 'gamg', fem_load_mode: str = 'riesz') -> None:
        if not np.isfinite([gap, rho, pressure_drop, ramp_time]).all() \
                or gap <= 0. or rho <= 0. or pressure_drop <= 0. or ramp_time < 0.:
            raise ValueError('Positive gap, rho, pressure_drop and nonnegative ramp_time required')
        if type(convection) is not bool:
            raise ValueError('convection must be a boolean')
        if inlet_treatment not in ('natural', 'energy_stable_pressure'):
            raise ValueError('Unknown inlet_treatment')
        self.inlet_treatment = inlet_treatment
        vb, _ = space.bases()
        mesh = vb.mesh
        boundaries = mesh.boundaries
        if boundaries is None or any(k not in boundaries for k in ('walls', 'inlet', 'outlet')):
            raise ValueError('Named walls, inlet and outlet boundaries required')
        fixed = np.concatenate([boundaries[k] for k in ('walls', 'airfoil') if k in boundaries])
        self.gap, self.rho = float(gap), float(rho)
        self.pressure_drop, self.ramp_time = float(pressure_drop), float(ramp_time)
        self.beta = CONVECTION_COEFFICIENT if convection else 0.
        self.drag = 10.*nu/gap**2
        super().__init__(space, nu, dirichlet_facets=fixed,
                         boundary_velocity=lambda x, y: (0., 0.),
                         reaction_coefficient=10./gap**2,
                         outlet_treatment=outlet_treatment, tolerance=tolerance,
                         cache_size=cache_size, stokes_backend=stokes_backend,
                         stokes_ordering=stokes_ordering, mass_backend=mass_backend,
                         stokes_amg_type=stokes_amg_type, fem_load_mode=fem_load_mode)
        # Backflow damping is an outlet option, not an artificial inlet resistance.
        self._open_mask = np.isin(self._all_facets.find, boundaries['outlet'])[:, None]
        self._inlet_mask = np.isin(self._all_facets.find, boundaries['inlet'])[:, None]
        inlet = FacetBasis(mesh, self.velocity_basis.elem, facets=boundaries['inlet'],
                           intorder=space.quadrature_order)

        @LinearForm
        def traction(v: Any, w: Any) -> Any:
            return -np.sum(v*w.n, axis=0)/(TIME_COEFFICIENT*rho)

        self._unit_pressure_load = np.asarray(asm(traction, inlet))
        unit = self._mass_factor.solve(self._unit_pressure_load[self.free_dofs])
        self.force: Callable[[float], Array] = lambda t: self.imposed_pressure(t)*unit

    def imposed_pressure(self, t: float) -> float:
        """Physical inlet pressure; outlet reservoir pressure is the zero reference."""
        if not np.isfinite(t) or t < 0.:
            raise ValueError('Finite nonnegative time required')
        s = 1. if self.ramp_time == 0. else min(1., t/self.ramp_time)
        return self.pressure_drop*s**3*(10.-15.*s+6.*s*s)

    def nonlinear_with_lifting(self, velocity: Array) -> tuple[Array, float]:
        if self.beta == 0.:
            return np.zeros_like(self._velocity(velocity)), 0.
        load, work = super().nonlinear_with_lifting(velocity)
        return self.beta*load, self.beta*work

    def _nonlinear_load_with_lifting(self, velocity: Array) -> tuple[Array, float]:
        if self.beta == 0.:
            return np.zeros_like(self._velocity(velocity)), 0.
        load, work = super()._nonlinear_load_with_lifting(velocity)
        return self.beta*load, self.beta*work

    def _boundary_transport(self, normal_velocity: Array) -> Array:
        transport = super()._boundary_transport(normal_velocity)
        if self.inlet_treatment == 'energy_stable_pressure':
            # Reservoir work supplies inflow kinetic energy. The negative
            # convective self-work must not amplify a free tangential trace.
            # Use the identical transport in momentum and the SAV pairing.
            return transport+np.maximum(-normal_velocity, 0.)*self._inlet_mask
        return transport

    def physical_pressure(self, pressure: Array) -> Array:
        """Convert an instantaneous normalized pressure, never a stage increment."""
        return TIME_COEFFICIENT*self.rho*self._pressure(pressure)

    def diagnostics(self, state: State) -> dict[str, float]:
        values = super().diagnostics(state)
        full = self.full_velocity(self.vector(state))
        values['convective_boundary_power'] *= self.beta
        if 'outlet_backflow_dissipation' in values:
            values['outlet_backflow_dissipation'] *= self.beta
        values.update(viscous_dissipation=self.nu*float(full @ (self.full_stiffness @ full)),
                      gap_dissipation=self.drag*float(full @ (self.full_mass @ full)),
                      pressure_power=self.imposed_pressure(state.t)*float(full @ self._unit_pressure_load),
                      profile_kinetic=TIME_COEFFICIENT*self.rho*self.gap*values['kinetic'])
        field, _ = self._boundary_quadrature.evaluate(full)
        normal = np.sum(field*self._all_facets.normals, axis=0)
        if self.inlet_treatment == 'energy_stable_pressure':
            values['inlet_kinetic_energy_correction'] = .5*self.beta*float(np.sum(
                np.maximum(-normal, 0.)*self._inlet_mask*np.sum(field**2, axis=0)*self._all_facets.dx))
        boundaries = self.velocity_basis.mesh.boundaries
        for name in ('inlet', 'outlet'):
            mask = np.isin(self._all_facets.find, boundaries[name])
            values[name+'_flow'] = float(np.sum(normal[mask]*self._all_facets.dx[mask]))
        height = float(np.sum(self._all_facets.dx[np.isin(self._all_facets.find, boundaries['outlet'])]))
        values['outlet_mean_velocity'] = values['outlet_flow']/height
        return values

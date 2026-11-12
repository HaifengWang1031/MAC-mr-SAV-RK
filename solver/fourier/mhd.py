"""Pressure-free, doubly periodic incompressible MHD in the Fourier model seam."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..core import Stage, State
from ..model import SolveResult
from .space import ComplexArray, FourierSpace, RealArray


@dataclass(frozen=True, kw_only=True)
class MHDState(State):
    """Real periodic samples with magnetic components in (2, ny, nx) order."""

    magnetic: RealArray


@dataclass
class MHDResponse:
    velocity: ComplexArray
    pressure: ComplexArray
    residual: float
    divergence_inf: float


class FourierMHD:
    """Paired (u, B) coefficients in (2, 2, ny, nx//2+1) order."""

    def __init__(self, space: FourierSpace, nu: float, eta: float, *,
                 tolerance: float = 1e-10) -> None:
        if not np.isfinite([nu, eta, tolerance]).all() or min(nu, eta, tolerance) <= 0:
            raise ValueError('MHD diffusivities and tolerance must be finite and positive')
        self.space = space
        self.nu = nu
        self.eta = eta
        self.tolerance = tolerance
        # The shared schemes multiply apply_K by model.nu.
        self._k = np.stack((space.k2, (eta/nu)*space.k2))[:, None]
        self._zero = np.zeros((2, 2, *space.spectral_shape), dtype=complex)
        self._zero.flags.writeable = False
        self.force: Callable[[float], ComplexArray] = self._evaluate_force
        # Empty multiplier payload for the shared scheme. There is no pressure block.
        self._no_pressure = np.empty(0, dtype=complex)

    def _evaluate_force(self, t: float) -> ComplexArray:
        return self._zero

    def vector(self, state: MHDState) -> ComplexArray:
        values = np.stack((np.stack((state.u, state.v)), state.magnetic))
        return self.space.to_spectral(values)

    def state(self, t: float, velocity: ComplexArray, r: float = 0.) -> MHDState:
        fields = self.space.to_physical(velocity)
        return MHDState(t, fields[0, 0], fields[0, 1], r, magnetic=fields[1])

    def combine(self, *terms: tuple[float, ComplexArray]) -> ComplexArray:
        total = np.zeros_like(terms[0][1], dtype=complex)
        for coefficient, value in terms:
            total += coefficient*value
        return total

    def apply_K(self, velocity: ComplexArray) -> ComplexArray:
        return self._k*velocity

    def apply_G(self, pressure: ComplexArray) -> ComplexArray:
        return self._zero

    def apply_D(self, velocity: ComplexArray) -> ComplexArray:
        return self.space.divergence(velocity)

    def inner(self, left: ComplexArray, right: ComplexArray) -> float:
        return self.space.inner(left, right)

    def max_abs(self, field: ComplexArray) -> float:
        """The eliminated pressure is an empty field with zero norm."""
        return float(np.max(np.abs(field), initial=0.))

    def physical_divergence_inf(self, velocity: ComplexArray) -> float:
        return float(np.max(np.abs(self.space.to_physical(self.apply_D(velocity)))))

    def stage_divergences(self, velocity: ComplexArray) -> tuple[float, float]:
        divergence = self.apply_D(velocity)
        physical = float(np.max(np.abs(self.space.to_physical(divergence))))
        modal = self.max_abs(divergence)
        if physical > self.tolerance*(1+self.max_abs(velocity)):
            raise RuntimeError(f'Fourier MHD joint divergence check failed: {physical}')
        return physical, modal

    def nonlinear(self, velocity: ComplexArray) -> ComplexArray:
        """Dealiased momentum flux and antisymmetric magnetic flux."""
        s = self.space
        (u, v), (bx, by) = s.to_physical(velocity)
        uu, uv, vv, emf = s.to_spectral(np.stack((
            u*u-bx*bx, u*v-bx*by, v*v-by*by, u*by-v*bx)))
        momentum = np.stack((1j*(s.kx*uu+s.ky*uv), 1j*(s.kx*uv+s.ky*vv)))
        induction = np.stack((-1j*s.ky*emf, 1j*s.kx*emf))
        return np.stack((s.project(momentum), induction))

    def nonlinear_with_lifting(self, velocity: ComplexArray) -> tuple[ComplexArray, float]:
        return self.nonlinear(velocity), 0.

    def solve(self, rhs: ComplexArray, *, mass: float, viscosity: float) -> MHDResponse:
        if not np.isfinite([mass, viscosity]).all() or mass <= 0 or viscosity < 0:
            raise ValueError('Projected MHD requires positive mass and nonnegative diffusion')
        if rhs.shape != self._zero.shape:
            raise ValueError('Expected both MHD Fourier vector fields')
        for block in rhs:
            self.space.check_velocity(block)
        diagonal = mass+viscosity*self._k
        velocity = rhs/diagonal
        residual = self.max_abs(diagonal*velocity-rhs)/(1+self.max_abs(rhs))
        divergence, _ = self.stage_divergences(velocity)
        if not np.isfinite(residual) or residual > self.tolerance:
            raise RuntimeError(f'Fourier MHD projected stage residual check failed: {residual}')
        return MHDResponse(velocity, self._no_pressure, residual, divergence)

    def solve_columns(self, columns: Sequence[ComplexArray], *, mass: float,
                      viscosity: float) -> list[SolveResult]:
        return [self.solve(rhs, mass=mass, viscosity=viscosity) for rhs in columns]

    def stage(self, pressure: ComplexArray, residual: float, divergence_inf: float,
              r: float = 0., candidates: Sequence[float] = (),
              root_residuals: Sequence[float] = (), scalar_residual: float = 0., *,
              continuity_residual: float = float('nan')) -> Stage:
        return Stage(self._no_pressure.real, residual, divergence_inf, r,
                     list(candidates), list(root_residuals), scalar_residual, continuity_residual)

    def diagnostics(self, state: MHDState) -> dict[str, float]:
        z = self.vector(state)
        s = self.space
        u, magnetic = z
        kinetic = .5*s.inner(u, u)/s.area
        magnetic_energy = .5*s.inner(magnetic, magnetic)/s.area
        dissipation = (self.nu*s.inner(u, s.k2*u)+self.eta*s.inner(magnetic, s.k2*magnetic))/s.area
        div_u, div_b = np.max(np.abs(s.to_physical(self.apply_D(z))), axis=(-2, -1))
        total = kinetic+magnetic_energy
        return {'kinetic': kinetic, 'magnetic': magnetic_energy, 'total_energy': total,
                'modified_energy': total+.5*(state.r-1)**2/s.area, 'dissipation': dissipation,
                'divergence_inf': float(div_u), 'magnetic_divergence_inf': float(div_b), 'r': state.r}

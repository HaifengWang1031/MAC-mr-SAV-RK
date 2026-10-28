"""Periodic Fourier model behind the unchanged SDIRK/mr-ccSAV model seam.

State and exported Stage pressure are real samples. Internal velocities, forces,
nonlinear loads and raw pressure multipliers are forward-normalized coefficients.
Loads remain unprojected so the full velocity-pressure stage equation is checked.
max_abs is a modal infinity norm; physical divergence uses inverse FFT separately.
"""
from collections.abc import Callable, Sequence

import numpy as np

from ..core import Stage, State
from ..model import SolveResult
from .space import ComplexArray, FourierSpace
from .stokes import FourierSolution, FourierStokes


class FourierModel:
    def __init__(self, space: FourierSpace, nu: float, *,
                 force: Callable[[float], ComplexArray] | None = None,
                 tolerance: float = 1e-10) -> None:
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError('nu must be positive and finite')
        self.space = space
        self.nu = nu
        self.backend = FourierStokes(space, tolerance)
        self._force = force
        self.force: Callable[[float], ComplexArray] = self._evaluate_force

    def zero_velocity(self) -> ComplexArray:
        return np.zeros((2, *self.space.spectral_shape), dtype=complex)

    def _evaluate_force(self, t: float) -> ComplexArray:
        return self.zero_velocity() if self._force is None else self.space.check_velocity(self._force(t))

    def vector(self, state: State) -> ComplexArray:
        return self.space.velocity(state.u, state.v, project=False)

    def state(self, t: float, velocity: ComplexArray, r: float = 0.) -> State:
        values = self.space.to_physical(self.space.check_velocity(velocity))
        return State(t, values[0], values[1], r)

    def combine(self, *terms: tuple[float, ComplexArray]) -> ComplexArray:
        total = np.zeros_like(terms[0][1], dtype=complex)
        for coefficient, value in terms:
            total += coefficient*value
        return total

    def apply_K(self, velocity: ComplexArray) -> ComplexArray:
        return self.space.k2*velocity

    def apply_G(self, pressure: ComplexArray) -> ComplexArray:
        return self.space.gradient(pressure)

    def apply_D(self, velocity: ComplexArray) -> ComplexArray:
        return self.space.divergence(velocity)

    def max_abs(self, field: ComplexArray) -> float:
        return float(np.max(np.abs(field)))

    def inner(self, left: ComplexArray, right: ComplexArray) -> float:
        return self.space.inner(left, right)

    def physical_divergence_inf(self, velocity: ComplexArray) -> float:
        return float(np.max(np.abs(self.space.to_physical(self.apply_D(velocity)))))

    def stage_divergences(self, velocity: ComplexArray) -> tuple[float, float]:
        return self.physical_divergence_inf(velocity), self.max_abs(self.apply_D(velocity))

    def nonlinear(self, velocity: ComplexArray) -> ComplexArray:
        """Strictly dealiased skew convection: half (advective + conservative).

        Input and product output use the same retained band. Even-grid Nyquist
        modes are absent, so the derivative is skew-adjoint on real samples.
        No Leray projection here: gradient loads must reach the pressure solve.
        """
        s = self.space
        z = s.check_velocity(velocity)
        values = s.to_physical(z)
        dx = s.to_physical(1j*s.kx*z)
        dy = s.to_physical(1j*s.ky*z)
        advective = s.to_spectral(values[0]*dx+values[1]*dy)
        u, v = values
        flux = s.to_spectral(np.stack([u*u, u*v, v*v]))
        conservative = np.stack([1j*(s.kx*flux[0]+s.ky*flux[1]),
                                 1j*(s.kx*flux[1]+s.ky*flux[2])])
        return .5*(advective+conservative)

    def nonlinear_with_lifting(self, velocity: ComplexArray) -> tuple[ComplexArray, float]:
        return self.nonlinear(velocity), 0.

    def solve(self, rhs: ComplexArray, *, mass: float, viscosity: float) -> FourierSolution:
        return self.backend.solve(rhs, mass=mass, viscosity=viscosity)

    def solve_columns(self, columns: Sequence[ComplexArray], *, mass: float,
                      viscosity: float) -> list[SolveResult]:
        return list(self.backend.solve_columns(columns, mass=mass, viscosity=viscosity))

    def stage(self, pressure: ComplexArray, residual: float, divergence_inf: float,
              r: float = 0., candidates: Sequence[float] = (),
              root_residuals: Sequence[float] = (), scalar_residual: float = 0., *,
              continuity_residual: float = float('nan')) -> Stage:
        return Stage(self.space.to_physical(pressure), residual, divergence_inf, r,
                     list(candidates), list(root_residuals), scalar_residual, continuity_residual)

    def diagnostics(self, state: State) -> dict[str, float]:
        z = self.vector(state)
        kinetic = .5*self.inner(z, z)
        h1 = self.inner(z, self.apply_K(z))
        curl = 1j*(self.space.kx*z[1]-self.space.ky*z[0])
        return {'kinetic': kinetic, 'modified_energy': kinetic+.5*(state.r-1)**2,
                'h1_seminorm_squared': h1, 'r': state.r,
                'divergence_inf': self.physical_divergence_inf(z),
                'mean_u': float(z[0, 0, 0].real), 'mean_v': float(z[1, 0, 0].real),
                'power': self.inner(self.force(state.t), z), 'dissipation': self.nu*h1,
                'enstrophy': .5*self.space.inner(curl, curl)}

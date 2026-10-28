"""Channel Model seam; all time schemes use the existing common drivers."""
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from ..core import Stage, State
from ..mac.grid import Array
from ..mac.stokes import DirectStokes, StokesBackend, StokesResult
from ..model import SolveResult
from .grid import ChannelGrid
from .kernels import convection, divergence
from .operators import ChannelOperators
from .fft_stokes import FFTStokes


class ChannelModel:
    def __init__(self, grid: ChannelGrid, nu: float,
                 force: Callable[[float], Array] | None = None,
                 backend: StokesBackend | Literal['direct', 'fft'] | None = None, cache_size: int = 4,
                 tolerance: float = 1e-10) -> None:
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError('nu must be positive and finite')
        self.grid, self.nu = grid, nu
        self.ops = ChannelOperators(grid)
        self.backend: StokesBackend
        if backend is None or backend == 'direct':
            self.backend = DirectStokes(self.ops, cache_size, tolerance)
        elif backend == 'fft':
            self.backend = FFTStokes(self.ops, cache_size, tolerance)
        elif isinstance(backend, str):
            raise ValueError('Unknown channel backend')
        else:
            self.backend = backend
        self.force = force if force is not None else lambda t: np.zeros(grid.size)

    def vector(self, state: State) -> Array:
        return self.grid.pack(state.u, state.v)

    def state(self, t: float, velocity: Array, r: float = 0.) -> State:
        u, v = self.grid.unpack(velocity)
        return State(t, u, v, r)

    def nonlinear(self, velocity: Array) -> Array:
        u, v = self.grid.unpack(velocity)
        cu, cv = convection(u, v, self.grid.hx, self.grid.hy)
        return self.grid.pack(cu, cv)

    def nonlinear_with_lifting(self, velocity: Array) -> tuple[Array, float]:
        return self.nonlinear(velocity), 0.

    def combine(self, *terms: tuple[float, Array]) -> Array:
        total = np.zeros_like(terms[0][1], dtype=float)
        for coefficient, field in terms:
            total += coefficient*field
        return total

    def apply_K(self, velocity: Array) -> Array:
        return self.ops.K@velocity

    def apply_G(self, pressure: Array) -> Array:
        return self.ops.G@np.asarray(pressure).reshape(-1)

    def apply_D(self, velocity: Array) -> Array:
        return self.ops.D@velocity

    def physical_divergence_inf(self, velocity: Array) -> float:
        u, v = self.grid.unpack(velocity)
        return self.max_abs(divergence(u, v, self.grid.hx, self.grid.hy))

    def stage_divergences(self, velocity: Array) -> tuple[float, float]:
        return self.physical_divergence_inf(velocity), self.max_abs(self.apply_D(velocity))

    def inner(self, left: Array, right: Array) -> float:
        return self.grid.inner(left, right)

    def max_abs(self, field: Array) -> float:
        return float(np.max(np.abs(field)))

    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> StokesResult:
        return self.backend.solve(rhs, mass=mass, viscosity=viscosity)

    def solve_columns(self, columns: Sequence[Array], *, mass: float,
                      viscosity: float) -> list[SolveResult]:
        solved = self.solve(np.column_stack(columns), mass=mass, viscosity=viscosity)
        return [StokesResult(solved.velocity[:, k], solved.pressure[:, :, k],
                            solved.residual, solved.divergence_inf) for k in range(len(columns))]

    def stage(self, pressure: Array, residual: float, divergence_inf: float, r: float = 0.,
              candidates: Sequence[float] = (), root_residuals: Sequence[float] = (),
              scalar_residual: float = 0., *, continuity_residual: float = float('nan')) -> Stage:
        return Stage(pressure, residual, divergence_inf, r, list(candidates),
                     list(root_residuals), scalar_residual, continuity_residual)

    def collocated_velocity(self, velocity: Array) -> tuple[Array, Array]:
        u, v = self.grid.unpack(velocity)
        return .5*(u[:, :-1]+u[:, 1:]), .5*(v[:-1]+v[1:])

    def flow_rates(self, velocity: Array) -> Array:
        u, _ = self.grid.unpack(velocity)
        return self.grid.hy*np.sum(u[:, :-1], axis=0)

    def diagnostics(self, state: State) -> dict[str, float]:
        w = self.vector(state)
        u, _ = self.grid.unpack(w)
        kinetic = .5*self.inner(w, w)
        h1 = self.inner(w, self.apply_K(w))
        flux = self.flow_rates(w)
        return {'kinetic': kinetic, 'modified_energy': kinetic+.5*(state.r-1)**2,
                'divergence_inf': self.physical_divergence_inf(w), 'r': state.r,
                'h1_seminorm_squared': h1, 'power': self.inner(self.force(state.t), w),
                'dissipation': self.nu*h1, 'bulk_velocity': float(np.mean(flux)/self.grid.ly),
                'flow_rate': float(np.mean(flux)), 'flow_rate_spread': float(np.ptp(flux)),
                'wall_shear_bottom': float(2*self.nu*np.mean(u[0, :-1])/self.grid.hy),
                'wall_shear_top': float(-2*self.nu*np.mean(u[-1, :-1])/self.grid.hy)}

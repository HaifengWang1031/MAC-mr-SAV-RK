"""Diagonal shifted Stokes solve with Leray projection and raw pressure multiplier."""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .space import ComplexArray, FourierSpace


@dataclass
class FourierSolution:
    velocity: ComplexArray
    pressure: ComplexArray
    residual: float
    divergence_inf: float


class FourierStokes:
    def __init__(self, space: FourierSpace, tolerance: float = 1e-10) -> None:
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Stokes tolerance must be finite and positive')
        self.space = space
        self.tolerance = tolerance
        self.rhs_count = 0

    def solve_columns(self, columns: Sequence[ComplexArray], *, mass: float,
                      viscosity: float) -> list[FourierSolution]:
        if not np.isfinite([mass, viscosity]).all() or min(mass, viscosity) < 0 \
                or mass+viscosity == 0:
            raise ValueError('Nonnegative, nonzero shifted Stokes coefficients required')
        if not columns:
            raise ValueError('At least one Stokes RHS required')
        s = self.space
        rhs = np.stack([s.check_velocity(column) for column in columns])
        if mass == 0 and np.max(np.abs(rhs[:, :, 0, 0])) > 1e-13*(1+np.max(np.abs(rhs))):
            raise ValueError('Steady periodic Stokes requires zero-mean forcing')
        pressure = -1j*(s.kx*rhs[:, 0]+s.ky*rhs[:, 1])*s.inverse_k2
        denominator = mass+viscosity*s.k2
        # Mean pressure is fixed to zero. A steady solve also fixes velocity mean;
        # positive-mass stages retain the mean RHS divided by mass.
        denominator = denominator.copy()
        if mass == 0:
            denominator[0, 0] = 1.
        velocity = (rhs-s.gradient(pressure))/denominator
        if mass == 0:
            velocity[:, :, 0, 0] = 0.
        defect = mass*velocity+viscosity*s.k2*velocity+s.gradient(pressure)-rhs
        divergence = s.divergence(velocity)
        results = []
        for index in range(len(columns)):
            residual = float(np.max(np.abs(defect[index]))/(1+np.max(np.abs(rhs[index]))))
            full_divergence = float(np.max(np.abs(s.to_physical(divergence[index]))))
            scale = 1+float(np.max(np.abs(s.to_physical(velocity[index]))))
            if not np.isfinite([residual, full_divergence]).all() \
                    or residual > self.tolerance or full_divergence > self.tolerance*scale:
                raise RuntimeError('Fourier Stokes momentum/divergence check failed')
            results.append(FourierSolution(velocity[index], pressure[index], residual, full_divergence))
        self.rhs_count += len(columns)
        return results

    def solve(self, rhs: ComplexArray, *, mass: float, viscosity: float) -> FourierSolution:
        return self.solve_columns([rhs], mass=mass, viscosity=viscosity)[0]

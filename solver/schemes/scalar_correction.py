"""Fixed-count frozen-factor SAV stage solves, sharing the two Stokes responses."""
from dataclasses import dataclass
import numpy as np

from ..core import Stage


@dataclass(frozen=True)
class ScalarCorrection:
    predictor: float
    values: tuple[float, ...]
    factor: float

    @property
    def r(self) -> float:
        return self.values[-1]

    @property
    def coupling(self) -> float:
        return self.factor*(1-self.r)

    def record(self, stage: Stage) -> Stage:
        stage.correction_predictor = self.predictor
        stage.corrections = list(self.values)
        stage.frozen_factor = self.factor
        return stage


def solve_scalar_stage(predictor: float, dt: float, d: float, rhs: float,
                       alpha: float, beta: float, *, order: int) -> ScalarCorrection:
    """Apply Phi exactly once for orders1/2 and exactly twice for order3.

    The returned factor is evaluated at the penultimate iterate. Reuse it in
    velocity, pressure and scalar residuals, including any known lifting work.
    """
    if order not in (1, 2, 3):
        raise ValueError('SAV order must be 1, 2 or 3')
    if not np.isfinite([predictor, dt, d, rhs, alpha, beta]).all() or dt <= 0 or d <= 0 or beta < 0:
        raise ValueError('Invalid frozen SAV scalar inputs')
    values = []
    z = predictor
    for _ in range(2 if order == 3 else 1):
        factor = 1. if order == 1 else (1+z if order == 2 else 1+z+z*z)
        response = dt*dt*factor*factor*beta
        denominator = d+response
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError('Invalid frozen SAV scalar denominator')
        z = (rhs-dt*factor*alpha+response)/denominator
        if not np.isfinite(z):
            raise RuntimeError('Nonfinite frozen SAV correction')
        values.append(float(z))
    return ScalarCorrection(float(predictor), tuple(values), float(factor))

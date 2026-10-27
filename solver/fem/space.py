"""Deterministic affine rectangular Taylor–Hood space and DOF maps."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray
from skfem import Basis, ElementTriP1, ElementTriP2, ElementVector, MeshTri

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Space:
    nx: int
    ny: int
    lx: float = 1.
    ly: float = 1.
    quadrature_order: int = 6

    def __post_init__(self) -> None:
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (self.nx, self.ny)):
            raise ValueError('nx and ny must be positive integers')
        if not np.isfinite([self.lx, self.ly]).all() or min(self.lx, self.ly) <= 0:
            raise ValueError('Rectangle lengths must be finite and positive')
        if not isinstance(self.quadrature_order, int) or self.quadrature_order < 5:
            raise ValueError('quadrature_order must be an integer >= 5 (P2 skew convection)')

    def bases(self) -> tuple[Any, Any]:
        mesh = MeshTri.init_tensor(np.linspace(0., self.lx, self.nx+1),
                                   np.linspace(0., self.ly, self.ny+1))
        return (Basis(mesh, ElementVector(ElementTriP2()), intorder=self.quadrature_order),
                Basis(mesh, ElementTriP1(), intorder=self.quadrature_order))

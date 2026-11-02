"""Independent scalar-pressure P2 Darcy reference on an explicitly supplied mesh."""
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementTriP2, FacetBasis, LinearForm, MeshTri, asm
from skfem.helpers import dot, grad


@BilinearForm
def _laplace(p: Any, q: Any, w: Any) -> Any:
    return dot(grad(p), grad(q))


@BilinearForm
def _mass(p: Any, q: Any, w: Any) -> Any:
    return p*q


@LinearForm
def _load(q: Any, w: Any) -> Any:
    return q*w.value_at


@dataclass
class DarcyResult:
    pressure: np.ndarray
    u: np.ndarray
    v: np.ndarray
    fluxes: dict[str, float]
    residual: float


def solve_darcy(mesh: MeshTri, nu: float, gap: float, pressure_drop: float, *,
                rho: float = 1., quadrature_order: int = 6) -> DarcyResult:
    """p_in=pressure_drop, p_out=0; other boundaries have weak no-through-flow.

    u/v are continuous P2 display projections of -gap**2 grad(p)/(12 rho nu).
    Reported fluxes use the actual pressure gradient, not the display projection.
    This scalar formulation is not a locally conservative mixed Darcy method.
    """
    if not np.isfinite([nu, gap, rho, pressure_drop]).all() or min(nu, gap, rho, pressure_drop) <= 0.:
        raise ValueError('Positive Darcy parameters required')
    if mesh.boundaries is None or any(k not in mesh.boundaries for k in ('inlet', 'outlet')):
        raise ValueError('Named inlet and outlet required')
    basis = Basis(mesh, ElementTriP2(), intorder=quadrature_order)
    inlet = basis.get_dofs(facets=mesh.boundaries['inlet']).all()
    outlet = basis.get_dofs(facets=mesh.boundaries['outlet']).all()
    if np.intersect1d(inlet, outlet).size:
        raise ValueError('Overlapping pressure ends')
    fixed = np.union1d(inlet, outlet)
    free = basis.complement_dofs(fixed)
    stiffness = asm(_laplace, basis).tocsc()
    p = np.zeros(basis.N)
    p[inlet] = pressure_drop
    rhs = -stiffness[free][:, fixed] @ p[fixed]
    p[free] = splu(stiffness[free][:, free].tocsc()).solve(rhs)
    residual = float(np.max(np.abs((stiffness @ p)[free]))/(1.+np.max(np.abs(rhs))))
    if not np.isfinite(p).all() or not np.isfinite(residual) or residual > 1e-8:
        raise RuntimeError('Darcy pressure solve failed the full residual check')
    mobility = gap**2/(12.*rho*nu)
    field = -mobility*basis.interpolate(p).grad
    mass = splu(asm(_mass, basis).tocsc())
    components = [mass.solve(asm(_load, basis, value_at=value)) for value in field]
    fluxes = {}
    for name, facets in mesh.boundaries.items():
        facet = FacetBasis(mesh, ElementTriP2(), facets=facets, intorder=quadrature_order)
        gradient = facet.interpolate(p).grad
        fluxes[name] = -mobility*float(np.sum(np.sum(gradient*facet.normals, axis=0)*facet.dx))
    return DarcyResult(p, components[0], components[1], fluxes, residual)

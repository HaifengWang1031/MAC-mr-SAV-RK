"""Read-only field sampling for the wedge cavity: vorticity, streamfunction, wall force.

Fields are returned at the mesh vertices, so a frame can be drawn with a single
`tripcolor(..., shading='gouraud')` / `tricontour` call instead of resampling onto
a Cartesian grid. That keeps every frame an exact nodal restriction of the P2
solution and avoids a point-location cost per frame.

The streamfunction is recovered from the discrete vorticity by a Poisson solve.
For a closed domain every wall is a streamline, so psi = 0 on the whole boundary
makes the Dirichlet problem unique and the recovered psi is the true one, not a
harmonic-function-shifted representative.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from skfem import Basis, BilinearForm, ElementTriP2, asm
from skfem.helpers import dot, grad

from solver.core import State
from solver.fem.cavity import ClosedCavityModel
from solver.fem.quadrature import P2Quadrature


@BilinearForm
def _laplace(u: Any, v: Any, w: Any) -> Any:
    return dot(grad(u), grad(v))


@BilinearForm
def _mass_scalar(u: Any, v: Any, w: Any) -> Any:
    return u*v


@dataclass
class Sample:
    """Nodal fields at the mesh vertices, plus the triangle connectivity."""
    x: np.ndarray
    y: np.ndarray
    triangles: np.ndarray
    speed: np.ndarray
    vorticity: np.ndarray
    streamfunction: np.ndarray
    pressure: np.ndarray | None = None


def streamline_levels(values: np.ndarray) -> np.ndarray:
    """Resolve each circulation sign separately, including a much weaker cell."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Finite nonempty streamfunction values required')
    low, high = float(np.min(values)), float(np.max(values))
    floor = 100*np.finfo(float).eps*max(1., abs(low), abs(high))
    fractions = np.geomspace(.025, .95, 9)
    negative = low*fractions[::-1] if low < -floor else np.empty(0)
    positive = high*fractions if high > floor else np.empty(0)
    return np.concatenate([negative, positive])


@dataclass
class FieldSampler:
    model: ClosedCavityModel
    quadrature_order: int | None = None
    _quadrature: P2Quadrature = field(init=False, repr=False)
    _scalar: Basis = field(init=False, repr=False)
    _psi_solve: Any = field(init=False, repr=False)
    _psi_free: np.ndarray = field(init=False, repr=False)
    _projection_solve: Any = field(init=False, repr=False)
    _vertices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        basis = self.model.velocity_basis
        order = self.quadrature_order if self.quadrature_order is not None \
            else self.model.space.quadrature_order
        self._quadrature = P2Quadrature(basis)
        # One P2 scalar space serves both the projection (mass) and the Poisson solve
        # (stiffness); the load is assembled with this basis's own test functions, so
        # the quadrature rule matches the velocity field's by construction.
        self._scalar = Basis(basis.mesh, ElementTriP2(), intorder=order)
        stiffness = asm(_laplace, self._scalar).tocsc()
        self._psi_free = self._scalar.complement_dofs(self._scalar.get_dofs().all())
        self._psi_solve = splu(stiffness[self._psi_free][:, self._psi_free].tocsc())
        self._projection_solve = splu(asm(_mass_scalar, self._scalar).tocsc())
        self._scatter = self._scalar.element_dofs
        self._phi = np.stack([np.asarray(function[0]) for function in self._scalar.basis])
        self._weights = self._scalar.dx
        # P2 nodal dofs at the vertices are the first N_nodes entries, in mesh.p order.
        self._vertex_dofs = self._scalar.nodal_dofs[0]
        if not np.allclose(self._scalar.doflocs[:, self._vertex_dofs], basis.mesh.p):
            raise RuntimeError('P2 vertex dofs do not follow the mesh vertex order')
        self._vertices = basis.mesh.p
        self._pressure_vertex_dofs = self.model.pressure_basis.nodal_dofs[0]

    def _scatter_load(self, values: np.ndarray) -> np.ndarray:
        local = np.einsum('eq,keq->ke', values*self._weights, self._phi)
        return np.bincount(self._scatter.ravel(), weights=local.ravel(), minlength=self._scalar.N)

    def vorticity(self, velocity: np.ndarray) -> np.ndarray:
        full = self.model.full_velocity(velocity)
        _, gradient = self._quadrature.evaluate(full)
        omega = gradient[1, 0]-gradient[0, 1]
        if omega.shape != self._weights.shape:
            raise RuntimeError('Vorticity and P2 test functions use different quadrature rules')
        return omega

    def streamfunction(self, velocity: np.ndarray) -> np.ndarray:
        """Full P2 streamfunction coefficients from the discrete vorticity."""
        load = self._scatter_load(self.vorticity(velocity))
        return self._streamfunction_from_load(load)

    def _streamfunction_from_load(self, load: np.ndarray) -> np.ndarray:
        psi = np.zeros(self._scalar.N)
        psi[self._psi_free] = self._psi_solve.solve(load[self._psi_free])
        return psi

    def sample(self, state: State, pressure: np.ndarray | None = None) -> Sample:
        velocity = self.model.vector(state)
        load = self._scatter_load(self.vorticity(velocity))
        omega = self._projection_solve.solve(load)
        psi = self._streamfunction_from_load(load)
        speed = np.hypot(state.u, state.v)
        speed_at = speed[self._vertex_dofs]
        vorticity_at = omega[self._vertex_dofs]
        psi_at = psi[self._vertex_dofs]
        pressure_at = None
        if pressure is not None:
            pressure = np.asarray(pressure, dtype=float)
            if pressure.shape != (self.model.pressure_basis.N,):
                raise ValueError('Invalid pressure coefficients for sampling')
            pressure_at = pressure[self._pressure_vertex_dofs]
        return Sample(x=self._vertices[0], y=self._vertices[1],
                      triangles=self.model.velocity_basis.mesh.t.T,
                      speed=speed_at, vorticity=vorticity_at, streamfunction=psi_at,
                      pressure=pressure_at)


def lid_force(model: ClosedCavityModel, state: State, pressure: np.ndarray,
              boundary: str = 'lid') -> np.ndarray:
    """Weak traction on a boundary: int (p I - nu grad u) n, n the solid's outward normal.

    For the horizontal sliding lid the omitted transposed-gradient term carries no
    x-component (no normal velocity varies along the wall), so the x-entry is the
    classical skin-friction drag.
    """
    return model.boundary_force(model.vector(state), pressure, boundary)


def vertex_extrema(sample: Sample, count: int = 6, *, radius_scale: float = 3.) -> list[dict[str, Any]]:
    """Local extrema of the nodal streamfunction, strongest first.

    Neighbours come from a k-d tree on the vertices, so the search is
    order-independent and no Cartesian grid is implied.
    """
    psi = sample.streamfunction
    points = np.column_stack([sample.x, sample.y])
    finite = np.isfinite(psi)
    tree = cKDTree(points[finite])
    spacing = float(np.median(tree.query(points[finite], k=2)[0][:, 1]))
    values = psi[finite]
    neighbours = tree.query_ball_tree(tree, radius_scale*spacing)
    candidates: list[tuple[float, int]] = []
    for index, near in enumerate(neighbours):
        window = values[near]
        centre = values[index]
        if abs(centre) < 1e-12 or not np.isfinite(centre):
            continue
        if centre != window.min() and centre != window.max():
            continue
        candidates.append((float(centre), index))
    candidates.sort(key=lambda item: -abs(item[0]))
    selected: list[dict[str, Any]] = []
    for value, index in candidates:
        point = points[finite][index]
        if any(np.hypot(*(point-np.array([item['x'], item['y']]))) < 2*radius_scale*spacing
               for item in selected):
            continue
        selected.append({'x': float(point[0]), 'y': float(point[1]), 'psi': value,
                         'kind': 'minimum' if value < 0 else 'maximum'})
        if len(selected) >= count:
            break
    return selected


def refine_extremum(sample: Sample, extremum: dict[str, Any], radius_scale: float = 2.5) -> dict[str, Any]:
    """Sub-vertex location of a streamfunction extremum from a local quadratic fit.

    The nodal argmin is quantised to the mesh, which is far too coarse to compare
    vortex positions between meshes. Fitting psi to a quadratic over the vertices
    within a few spacings and solving for its stationary point recovers the
    location to well below one element.
    """
    points = np.column_stack([sample.x, sample.y])
    finite = np.isfinite(sample.streamfunction)
    tree = cKDTree(points[finite])
    spacing = float(np.median(tree.query(points[finite], k=2)[0][:, 1]))
    centre = np.array([extremum['x'], extremum['y']])
    mask = finite & (np.hypot(sample.x-centre[0], sample.y-centre[1]) <= radius_scale*spacing)
    if np.count_nonzero(mask) < 6:
        return dict(extremum)
    dx, dy = sample.x[mask]-centre[0], sample.y[mask]-centre[1]
    design = np.column_stack([np.ones(dx.size), dx, dy, dx*dx, dx*dy, dy*dy])
    coefficients, *_ = np.linalg.lstsq(design, sample.streamfunction[mask], rcond=None)
    gradient = np.array([coefficients[1], coefficients[2]])
    hessian = np.array([[2.*coefficients[3], coefficients[4]],
                        [coefficients[4], 2.*coefficients[5]]])
    determinant = float(np.linalg.det(hessian))
    if abs(determinant) < 1e-18:
        return dict(extremum)
    step = -np.linalg.solve(hessian, gradient)
    if not np.all(np.isfinite(step)) or np.hypot(*step) > radius_scale*spacing:
        return dict(extremum)
    value = float(coefficients[0]+.5*gradient @ step)
    return {'x': float(centre[0]+step[0]), 'y': float(centre[1]+step[1]), 'psi': value,
            'kind': extremum['kind']}

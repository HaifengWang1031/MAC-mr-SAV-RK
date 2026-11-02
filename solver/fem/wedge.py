"""Lid-driven wedge cavity: an isosceles triangle closed by a sliding top lid.

Geometry, with lid length `L` and full apex angle `2*alpha`:

    lid    y = 0,   -W <= x <= W,   W = L/2
    walls  (W, 0) -- (0, -H) -- (-W, 0),   H = W/tan(alpha)

The fluid occupies y <= 0 only. The lid is the moving boundary; both inclined
walls are stationary no-slip. The apex (0, -H) is a corner where two no-slip
walls meet, so it carries the same corner singularity as a cavity corner.

The mesh is a structured right-triangle lattice mirrored about the symmetry
axis. Node (i, j) sits at (i*h, -j*k) with |i| + j <= n, so the lattice is
aligned with the apex and every triangle is the same right triangle with legs
h = W/n and k = H/n. Nothing degenerates at the apex: the last ring is a single
triangle with the same legs as every other cell. Element angles are 90 degrees,
atan(k/h) and 90 - atan(k/h), so apex angles in [60, 120] degrees stay inside
[30, 60] degrees on the acute corners. Every triangle is affine by construction,
which is what MeshSpace requires.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skfem import Basis, ElementTriP1, ElementTriP2, ElementVector, MeshTri

from .space import Array, MeshSpace

LID_LENGTH = 1.


def mesh_parameters(lid_length: float, apex_angle: float, lid_cells: int) -> tuple[float, float, float, int]:
    """Half apex angle, half lid width, height, and cells per half of the lid."""
    if not np.isfinite([lid_length, apex_angle]).all() or lid_length <= 0.:
        raise ValueError('Finite positive lid length required')
    if not 10. <= apex_angle < 170.:
        raise ValueError('apex_angle must lie in [10, 170) degrees')
    if isinstance(lid_cells, bool) or not isinstance(lid_cells, int) or lid_cells < 4 or lid_cells % 2:
        raise ValueError('lid_cells must be an even integer >= 4')
    alpha = np.deg2rad(apex_angle)/2.
    half_width = lid_length/2.
    height = half_width/np.tan(alpha)
    return float(alpha), float(half_width), float(height), lid_cells//2


def _mesh(lid_length: float, apex_angle: float, lid_cells: int) -> MeshTri:
    alpha, half_width, height, n = mesh_parameters(lid_length, apex_angle, lid_cells)
    hx, hy = half_width/n, height/n

    index: dict[tuple[int, int], int] = {}
    points: list[tuple[float, float]] = []
    for j in range(n + 1):
        for i in range(-(n - j), n - j + 1):
            index[(i, j)] = len(points)
            points.append((i*hx, -j*hy))

    def tri(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> list[tuple[int, int, int]]:
        return [(index[a], index[b], index[c]), (index[(-a[0], a[1])], index[(-c[0], c[1])],
                                                 index[(-b[0], b[1])])]

    triangles: list[tuple[int, int, int]] = []
    for j in range(n):
        for i in range(n - j):
            triangles += tri((i, j), (i, j + 1), (i + 1, j))
            if i + j + 2 <= n:
                triangles += tri((i + 1, j), (i, j + 1), (i + 1, j + 1))

    p = np.array(points, dtype=float).T
    t = np.array(triangles, dtype=np.int64).T
    corners = p[:, t]
    signed = .5*((corners[0, 1] - corners[0, 0])*(corners[1, 2] - corners[1, 0]) -
                 (corners[1, 1] - corners[1, 0])*(corners[0, 2] - corners[0, 0]))
    flip = signed < 0.
    t[:, flip] = t[[0, 2, 1]][:, flip]
    return validate_wedge_mesh(MeshTri(p.copy(), t.copy()), lid_length, apex_angle)


def validate_wedge_mesh(mesh: MeshTri, lid_length: float, apex_angle: float) -> MeshTri:
    """Classify the boundary and reject any mesh that is not the declared triangle."""
    MeshSpace(mesh)
    _, half_width, height, _ = mesh_parameters(lid_length, apex_angle, 4)
    x, y = mesh.p
    if np.any(y > 1e-12) or np.any(y < -height - 1e-12) \
            or np.any(np.abs(x) > half_width*(1. + y/height) + 1e-12):
        raise ValueError('Wedge vertices leave the declared triangle')

    facets = mesh.boundary_facets()
    ends = mesh.p[:, mesh.facets[:, facets]]
    slope = height/half_width
    masks = {'lid': np.all(np.isclose(ends[1], 0., rtol=0., atol=1e-12), axis=0),
             'wall_left': np.all(np.isclose(ends[1] + (half_width + ends[0])*slope, 0.,
                                            rtol=0., atol=1e-12), axis=0),
             'wall_right': np.all(np.isclose(ends[1] + (half_width - ends[0])*slope, 0.,
                                             rtol=0., atol=1e-12), axis=0)}
    if not np.all(np.sum(list(masks.values()), axis=0) == 1):
        raise RuntimeError('Unclassified or multiply classified wedge boundary')
    if not all(mask.any() for mask in masks.values()):
        raise RuntimeError('Empty wedge boundary group')

    corners = mesh.p[:, mesh.t]
    area = .5*np.abs((corners[0, 1] - corners[0, 0])*(corners[1, 2] - corners[1, 0]) -
                     (corners[1, 1] - corners[1, 0])*(corners[0, 2] - corners[0, 0]))
    if np.any(area <= 0.):
        raise RuntimeError('Degenerate wedge triangle')
    if abs(float(np.sum(area)) - half_width*height) > 1e-12*max(1., half_width*height):
        raise RuntimeError('Wedge mesh area does not match the declared triangle')
    if mesh.p.shape[1] - mesh.facets.shape[1] + mesh.t.shape[1] != 1:
        raise RuntimeError('Wedge mesh is not a simply connected disk')

    mesh = mesh.with_boundaries({name: facets[mask] for name, mask in masks.items()})
    assert mesh.boundaries is not None
    return mesh


def triangle_quality(mesh: MeshTri) -> dict[str, float]:
    """Smallest angle, longest/shortest edge ratio and area spread over all triangles."""
    p = mesh.p[:, mesh.t]
    e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=0)
    e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=0)
    e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=0)
    at_p0 = np.arccos(np.clip((e0**2 + e2**2 - e1**2)/(2*e0*e2), -1., 1.))
    at_p1 = np.arccos(np.clip((e0**2 + e1**2 - e2**2)/(2*e0*e1), -1., 1.))
    angles = np.stack([at_p0, at_p1, np.pi - at_p0 - at_p1])
    area = .5*e0*e2*np.sin(at_p0)
    return {'min_angle_deg': float(np.rad2deg(angles.min())),
            'max_aspect': float((np.maximum(np.maximum(e0, e1), e2) /
                                 np.minimum(np.minimum(e0, e1), e2)).max()),
            'min_area': float(area.min()), 'max_area': float(area.max())}


@dataclass(frozen=True)
class WedgeSpace:
    """P2/P1 Taylor–Hood space on the lid-driven wedge cavity."""
    lid_length: float = LID_LENGTH
    apex_angle: float = 90.
    lid_cells: int = 64
    quadrature_order: int = 6
    mesh: MeshTri = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.quadrature_order) is not int or self.quadrature_order < 5:
            raise ValueError('quadrature_order must be an integer >= 5')
        mesh = _mesh(self.lid_length, self.apex_angle, self.lid_cells)
        object.__setattr__(self, 'mesh', mesh)
        MeshSpace(mesh, self.quadrature_order)

    @property
    def height(self) -> float:
        return mesh_parameters(self.lid_length, self.apex_angle, self.lid_cells)[2]

    @property
    def half_width(self) -> float:
        return mesh_parameters(self.lid_length, self.apex_angle, self.lid_cells)[1]

    @property
    def element_size(self) -> tuple[float, float]:
        _, half_width, height, n = mesh_parameters(self.lid_length, self.apex_angle, self.lid_cells)
        return half_width/n, height/n

    def bases(self) -> tuple[Any, Any]:
        return (Basis(self.mesh, ElementVector(ElementTriP2()), intorder=self.quadrature_order),
                Basis(self.mesh, ElementTriP1(), intorder=self.quadrature_order))

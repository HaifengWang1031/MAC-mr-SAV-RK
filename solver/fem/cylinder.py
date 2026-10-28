"""DFG2D-2 geometry: a verified conforming polygonal hole with graded wake mesh."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay
from skfem import MeshTri

from .space import MeshSpace

LENGTH = 2.2
HEIGHT = .41
CENTER = np.array([.2, .2])
RADIUS = .05


def mesh_parameters(resolution: float, circle_segments: int | None) -> tuple[float, int]:
    if isinstance(resolution, bool) or not isinstance(resolution, (int, float, np.integer, np.floating)) \
            or not np.isfinite(resolution) or resolution < .25:
        raise ValueError('Cylinder resolution must be finite and >= .25')
    segments = max(16, 4*int(np.ceil(24*resolution))) if circle_segments is None else circle_segments
    if type(segments) is not int or segments < 16 or segments % 4:
        raise ValueError('circle_segments must be a multiple of four >= 16')
    return float(resolution), segments


def _mesh(resolution: float, segments: int) -> MeshTri:
    nx, ny = int(np.ceil(LENGTH*resolution/.08)), int(np.ceil(HEIGHT*resolution/.08))
    x, y = np.linspace(0., LENGTH, 8*nx+1), np.linspace(0., HEIGHT, 8*ny+1)

    def grid(stride: int) -> np.ndarray:
        xx, yy = np.meshgrid(x[::stride], y[::stride])
        return np.column_stack((xx.ravel(), yy.ravel()))

    coarse, wake, near = grid(8), grid(2), grid(1)
    wake = wake[(wake[:, 0] >= .2) & (wake[:, 1] >= .075) & (wake[:, 1] <= .335)]
    near = near[(near[:, 0] >= .075) & (near[:, 0] <= .4) &
                (near[:, 1] >= .075) & (near[:, 1] <= .325)]
    points = np.unique(np.vstack((coarse, wake, near)), axis=0)
    # Four first-layer intervals across D/sqrt(Re)≈.01 at the unit benchmark resolution.
    h = .0025/resolution
    points = points[np.linalg.norm(points-CENTER, axis=1) > RADIUS+4.6*h]
    theta = 2*np.pi*np.arange(segments)/segments
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    points = np.vstack([points, *[CENTER+(RADIUS+k*h)*directions for k in range(5)]])
    if np.any(points < 0.) or np.any(points[:, 0] > LENGTH) or np.any(points[:, 1] > HEIGHT):
        raise ValueError('Cylinder refinement rings leave the channel')
    triangles = Delaunay(points).simplices
    keep = np.linalg.norm(points[triangles].mean(axis=1)-CENTER, axis=1) > RADIUS
    triangles = triangles[keep]
    used = np.unique(triangles)
    indices = np.full(len(points), -1, dtype=np.int32)
    indices[used] = np.arange(len(used), dtype=np.int32)
    mesh = MeshTri(points[used].T, indices[triangles].T)
    return validate_cylinder_mesh(mesh, segments)


def validate_cylinder_mesh(mesh: MeshTri, segments: int) -> MeshTri:
    """Validate a saved mesh against the circular-polygon domain, without regenerating topology."""
    MeshSpace(mesh)
    if np.any(mesh.p[0] < -1e-12) or np.any(mesh.p[0] > LENGTH+1e-12) \
            or np.any(mesh.p[1] < -1e-12) or np.any(mesh.p[1] > HEIGHT+1e-12) \
            or np.any(np.linalg.norm(mesh.p-CENTER[:, None], axis=0) < RADIUS-1e-12):
        raise ValueError('Cylinder vertices leave the declared fluid domain')
    facets = mesh.boundary_facets()
    ends = mesh.p[:, mesh.facets[:, facets]]
    masks = {'inlet': np.all(np.isclose(ends[0], 0., rtol=0., atol=1e-12), axis=0),
             'outlet': np.all(np.isclose(ends[0], LENGTH, rtol=0., atol=1e-12), axis=0),
             'walls': np.all(np.isclose(ends[1], 0., rtol=0., atol=1e-12), axis=0) |
                      np.all(np.isclose(ends[1], HEIGHT, rtol=0., atol=1e-12), axis=0),
             'cylinder': np.all(np.isclose(np.linalg.norm(ends-CENTER[:, None, None], axis=0),
                                           RADIUS, rtol=0., atol=2e-12), axis=0)}
    if not np.all(np.sum(list(masks.values()), axis=0) == 1):
        raise RuntimeError('Unclassified or multiply classified cylinder boundary')
    mesh = mesh.with_boundaries({name: facets[mask] for name, mask in masks.items()})
    assert mesh.boundaries is not None
    obstacle = mesh.facets[:, mesh.boundaries['cylinder']]
    nodes = np.unique(obstacle)
    if len(nodes) != segments or obstacle.shape[1] != segments:
        raise RuntimeError('Circle polygon edges were not retained by triangulation')
    angles = np.arctan2(mesh.p[1, nodes]-.2, mesh.p[0, nodes]-.2)
    nodes = nodes[np.argsort(angles)]
    actual_edges = {tuple(sorted(edge)) for edge in obstacle.T}
    if any(tuple(sorted((int(nodes[k]), int(nodes[(k+1) % segments])))) not in actual_edges
           for k in range(segments)):
        raise RuntimeError('Cylinder boundary is not the intended simple polygon')
    corners = mesh.p[:, mesh.t]
    area = .5*np.abs((corners[0, 1]-corners[0, 0])*(corners[1, 2]-corners[1, 0])-
                     (corners[1, 1]-corners[1, 0])*(corners[0, 2]-corners[0, 0]))
    expected = LENGTH*HEIGHT-.5*segments*RADIUS**2*np.sin(2*np.pi/segments)
    adjacency = sp.csr_matrix((np.ones(mesh.facets.shape[1]), mesh.facets),
                               shape=(mesh.p.shape[1], mesh.p.shape[1]))
    components = connected_components(adjacency+adjacency.T, directed=False, return_labels=False)
    # An empty circumcircle makes these polygon edges Delaunay; verify, never assume a generic hole.
    if np.any(area <= 0.) or abs(float(np.sum(area))-expected) > 5e-12 or components != 1 \
            or mesh.p.shape[1]-mesh.facets.shape[1]+mesh.t.shape[1] != 0:
        raise RuntimeError('Invalid cylinder mesh area, connectivity or hole topology')
    return mesh


@dataclass(frozen=True)
class CylinderSpace:
    resolution: float = 1.
    circle_segments: int | None = None
    quadrature_order: int = 6
    mesh: MeshTri = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        resolution, segments = mesh_parameters(self.resolution, self.circle_segments)
        if type(self.quadrature_order) is not int or self.quadrature_order < 5:
            raise ValueError('quadrature_order must be an integer >= 5')
        object.__setattr__(self, 'resolution', resolution)
        object.__setattr__(self, 'circle_segments', segments)
        object.__setattr__(self, 'mesh', _mesh(self.resolution, segments))
        MeshSpace(self.mesh, self.quadrature_order)

    @property
    def sagitta(self) -> float:
        assert self.circle_segments is not None
        return float(RADIUS*(1-np.cos(np.pi/self.circle_segments)))

    def bases(self) -> tuple[Any, Any]:
        return MeshSpace(self.mesh, self.quadrature_order).bases()

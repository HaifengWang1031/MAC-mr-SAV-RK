"""Closed-TE NACA0012 geometry, optional Gmsh meshing, saved-mesh validation."""
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skfem import MeshTri


def airfoil_polygon(chord: float = 1., angle: float = 10., surface_points: int = 120) -> np.ndarray:
    """CCW polygon; nose-up angle rotates LE-to-TE clockwise about quarter chord."""
    if not np.isfinite([chord, angle]).all() or chord <= 0. or abs(angle) > 45. \
            or type(surface_points) is not int or surface_points < 16:
        raise ValueError('Positive chord, |angle|<=45, surface_points>=16 required')
    # Cosine spacing resolves curvature at both ends independently of fluid h.
    x = .5*(1.-np.cos(np.linspace(0., np.pi, surface_points//2+1)))
    y = .6*(.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
    y[[0, -1]] = 0.
    p = np.vstack((np.column_stack((x[::-1], y[::-1])), np.column_stack((x[1:-1], -y[1:-1]))))
    theta = -np.deg2rad(angle)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return chord*((p-[.25, 0.]) @ r.T+[.25, 0.])


def validate_airfoil_mesh(mesh: MeshTri, polygon: np.ndarray,
                          bounds: tuple[float, float, float, float]) -> dict[str, float]:
    """Check every exterior edge, a single connected hole, and polygonal area."""
    from matplotlib.path import Path
    xmin, xmax, ymin, ymax = bounds
    p, t = mesh.p.T, mesh.t.T
    if not np.isfinite(p).all() or len(t) == 0 or mesh.boundaries is None:
        raise ValueError('Invalid airfoil mesh')
    groups = mesh.boundaries
    if set(groups) != {'inlet', 'outlet', 'walls', 'airfoil'}:
        raise ValueError('Airfoil boundary labels missing')
    actual = np.concatenate(list(groups.values()))
    if len(np.unique(actual)) != len(actual) or not np.array_equal(np.sort(actual), mesh.boundary_facets()):
        raise ValueError('Unclassified or overlapping exterior edges')
    if np.any(p[:, 0] < xmin-1e-12) or np.any(p[:, 0] > xmax+1e-12) \
            or np.any(p[:, 1] < ymin-1e-12) or np.any(p[:, 1] > ymax+1e-12):
        raise ValueError('Mesh leaves outer rectangle')
    for name, coordinate, value in [('inlet', 0, xmin), ('outlet', 0, xmax)]:
        if not np.allclose(p[mesh.facets[:, groups[name]], coordinate], value, rtol=0, atol=1e-11):
            raise ValueError('Incorrect pressure-end label')
    w = p[mesh.facets[:, groups['walls']], 1]
    if not np.all(np.all(np.isclose(w, ymin, atol=1e-11), axis=0)
                  | np.all(np.isclose(w, ymax, atol=1e-11), axis=0)):
        raise ValueError('Incorrect lateral wall label')
    corners = p[t]
    first, second = corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0]
    cross = np.abs(first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0])
    hole_area = .5*abs(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                             -polygon[:, 1]*np.roll(polygon[:, 0], -1)))
    area_error = abs(float(np.sum(cross)*.5)-((xmax-xmin)*(ymax-ymin)-hole_area))
    if np.any(cross <= 0.) or area_error > 1e-9*max(1., (xmax-xmin)*(ymax-ymin)) \
            or np.any(Path(np.vstack((polygon, polygon[0]))).contains_points(corners.mean(axis=1))):
        raise ValueError('Wrong polygonal area, degenerate or solid-interior triangle')
    edges = mesh.facets
    graph = coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(len(p), len(p)))
    if connected_components(graph, directed=False)[0] != 1 or len(p)-edges.shape[1]+len(t) != 0:
        raise ValueError('Expected one connected fluid region with one hole')
    # Each saved foil edge must lie on one of the declared straight surface segments.
    midpoint = p[mesh.facets[:, groups['airfoil']]].mean(axis=0)
    distance = np.full(len(midpoint), np.inf)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        ab = b-a
        s = np.clip((midpoint-a) @ ab/(ab @ ab), 0., 1.)
        distance = np.minimum(distance, np.linalg.norm(midpoint-a-s[:, None]*ab, axis=1))
    if np.max(distance) > 1e-10*max(1., xmax-xmin):
        raise ValueError('Saved foil edges leave declared polygon')
    lengths = np.linalg.norm(corners-np.roll(corners, -1, axis=1), axis=2)
    minimum_angle = np.pi
    for i in range(3):
        a, b, c = lengths[:, i], lengths[:, (i+1) % 3], lengths[:, (i+2) % 3]
        minimum_angle = min(minimum_angle, float(np.min(np.arccos(np.clip((a*a+b*b-c*c)/(2*a*b), -1., 1.)))))
    return {'vertices': float(len(p)), 'triangles': float(len(t)),
            'area_error': area_error, 'polygonal_area': hole_area,
            'min_angle_deg': float(np.rad2deg(minimum_angle))}


def make_airfoil_mesh(*, chord: float = 1., angle: float = 10., gap_ratio: float = .02,
                      surface_points: int = 120, far_size: float = .4,
                      layer_cells: int = 4,
                      bounds: tuple[float, float, float, float] = (-3., 6., -2., 2.),
                      wake_region: tuple[float, float, float, float] | None = None,
                      wake_size: float | None = None) -> MeshTri:
    """Gmsh Distance/MathEval grading to h_min=gap/(sqrt(10)*layer_cells).

    Bounds and far_size are in chord units. Gmsh is needed only here; saved-run
    readers reconstruct MeshTri directly. No curved/high-order geometry is used.
    """
    import gmsh
    if not np.isfinite([gap_ratio, far_size, *bounds]).all() or min(gap_ratio, far_size) <= 0. \
            or type(layer_cells) is not int or layer_cells < 1:
        raise ValueError('Positive gap_ratio, far_size and integer layer_cells required')
    polygon = airfoil_polygon(chord, angle, surface_points)
    if (wake_region is None) != (wake_size is None):
        raise ValueError('wake_region and wake_size must be provided together')
    if wake_region is not None:
        region = np.asarray(wake_region, dtype=float)
        if region.shape != (4,) or not np.isfinite(region).all() or not region[0] < region[1] \
                or not region[2] < region[3] or wake_size is None or not np.isfinite(wake_size) or wake_size <= 0.:
            raise ValueError('Invalid wake refinement')
    rectangle = np.asarray(bounds)*chord
    xmin, xmax, ymin, ymax = rectangle
    if not xmin < np.min(polygon[:, 0]) < np.max(polygon[:, 0]) < xmax \
            or not ymin < np.min(polygon[:, 1]) < np.max(polygon[:, 1]) < ymax:
        raise ValueError('Foil must lie strictly inside rectangle')
    if gmsh.isInitialized():
        raise RuntimeError('Airfoil generation requires an unused Gmsh session')
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add('hele_shaw_airfoil')
        geo = gmsh.model.geo
        outer = [geo.addPoint(float(x), float(y), 0.) for x, y in
                 [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]]
        lines = [geo.addLine(outer[i], outer[(i+1) % 4]) for i in range(4)]
        foil = [geo.addPoint(float(x), float(y), 0.) for x, y in polygon]
        foil_lines = [geo.addLine(foil[i], foil[(i+1) % len(foil)]) for i in range(len(foil))]
        geo.addPlaneSurface([geo.addCurveLoop(lines), geo.addCurveLoop(foil_lines)])
        geo.synchronize()
        groups = {'inlet': [lines[3]], 'outlet': [lines[1]], 'walls': [lines[0], lines[2]],
                  'airfoil': foil_lines}
        distance = gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(distance, 'CurvesList', foil_lines)
        gmsh.model.mesh.field.setNumber(distance, 'Sampling', 100)
        size = gmsh.model.mesh.field.add('MathEval')
        hmin = chord*gap_ratio/(np.sqrt(10)*layer_cells)
        # Analytic wall distance avoids sampling a long wall more coarsely than b.
        gmsh.model.mesh.field.setString(size, 'F',
            f'Min({chord*far_size},{hmin}+0.35*Min(F{distance},Min(y-({ymin}),({ymax})-y)))')
        if wake_region is not None:
            assert wake_size is not None
            wake = gmsh.model.mesh.field.add('Box')
            for key, value in zip(('XMin', 'XMax', 'YMin', 'YMax'), wake_region):
                gmsh.model.mesh.field.setNumber(wake, key, value*chord)
            gmsh.model.mesh.field.setNumber(wake, 'VIn', wake_size*chord)
            gmsh.model.mesh.field.setNumber(wake, 'VOut', far_size*chord)
            gmsh.model.mesh.field.setNumber(wake, 'Thickness', .25*chord)
            combined = gmsh.model.mesh.field.add('Min')
            gmsh.model.mesh.field.setNumbers(combined, 'FieldsList', [size, wake])
            gmsh.model.mesh.field.setAsBackgroundMesh(combined)
        else:
            gmsh.model.mesh.field.setAsBackgroundMesh(size)
        for key in ['Mesh.MeshSizeExtendFromBoundary', 'Mesh.MeshSizeFromPoints', 'Mesh.MeshSizeFromCurvature']:
            gmsh.option.setNumber(key, 0)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        tags, p = tags[order], np.asarray(coordinates).reshape(-1, 3)[order, :2]
        _, connectivity = gmsh.model.mesh.getElementsByType(2)
        t = np.searchsorted(tags, np.asarray(connectivity).reshape(-1, 3))
        used = np.unique(t)
        remap = np.full(len(p), -1, dtype=int)
        remap[used] = np.arange(len(used))
        mesh = MeshTri(p[used].T.copy(), remap[t].T.copy())
        facet_map = {tuple(sorted(edge)): i for i, edge in enumerate(mesh.facets.T)}
        boundary_groups = {}
        for name, curves in groups.items():
            segments: list[int] = []
            for curve in curves:
                _, nodes = gmsh.model.mesh.getElementsByType(1, curve)
                indices = remap[np.searchsorted(tags, np.asarray(nodes).reshape(-1, 2))]
                segments.extend(facet_map[tuple(sorted(edge))] for edge in indices)
            boundary_groups[name] = np.sort(np.asarray(segments, dtype=np.int32))
        mesh = mesh.with_boundaries(boundary_groups)
    finally:
        gmsh.finalize()
    validate_airfoil_mesh(mesh, polygon, tuple(rectangle))
    return mesh

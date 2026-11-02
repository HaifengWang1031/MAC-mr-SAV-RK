"""Benchmark fixtures; imported after selecting a frozen solver source."""
from pathlib import Path
from typing import Any

import numpy as np
from skfem import MeshTri

from solver.fem import MeshSpace
from solver.fem.cavity import ClosedCavityModel
from solver.fem.hele_shaw import HeleShawModel
from solver.fem.open_flow import OpenFlowModel


def make_mesh(spec: dict[str, Any]) -> Any:
    family, level = spec['family'], spec['level']
    if family == 'cavity':
        n = spec['n']
        return MeshTri.init_tensor(np.linspace(0., 1., n+1), np.linspace(0., 1., n+1)).with_boundaries({
            'lid': lambda x: np.isclose(x[1], 1.)})
    if family == 'wedge':
        from solver.fem.wedge import WedgeSpace
        return WedgeSpace(apex_angle=25., lid_cells=spec['n']).mesh
    if family == 'cylinder':
        from solver.fem.cylinder import CylinderSpace
        return CylinderSpace(resolution=spec['resolution'], circle_segments=192).mesh
    from solver.fem.airfoil import make_airfoil_mesh
    return make_airfoil_mesh(gap_ratio=.02, surface_points=120, layer_cells=level+1,
                             far_size=.4, wake_region=(.5, 5., -.7, .5), wake_size=.1/(2**level))


def load_mesh(path: Path) -> Any:
    with np.load(path) as data:
        boundaries = {k[9:]: data[k] for k in data.files if k.startswith('boundary_')}
        return MeshTri(data['p'], data['t']).with_boundaries(boundaries)


def build_model(mesh: Any, spec: dict[str, Any], options: dict[str, Any]) -> Any:
    space = MeshSpace(mesh)
    if spec['family'] in ('cavity', 'wedge'):
        return ClosedCavityModel(space, .01, **options)
    if spec['family'] == 'cylinder':
        fixed = np.concatenate([mesh.boundaries[k] for k in ('inlet', 'walls', 'cylinder')])
        return OpenFlowModel(space, .0002, dirichlet_facets=fixed,
            boundary_velocity=lambda x, y: (np.where(np.isclose(x, 0.), 6*y*(.41-y)/.41**2, 0.), 0.),
            outlet_treatment='backflow_damped', **options)
    return HeleShawModel(space, .0001, .02, pressure_drop=27., ramp_time=1.,
        inlet_treatment='energy_stable_pressure', outlet_treatment='backflow_damped', **options)

"""Saved tagged cylinder mesh/coefficients; no dynamical-model construction on read."""
from importlib.metadata import version
from typing import Any

import numpy as np
from skfem import MeshTri

from experiments.fem_ns.representation import METADATA as FEM_METADATA, _array, maps, validate_fields
from solver.fem import MeshSpace, OpenFlowModel
from solver.fem.cylinder import CENTER, HEIGHT, LENGTH, RADIUS, validate_cylinder_mesh
from solver.integrate import Result
from .observations import FIELDS, PROBES, Observations
from .problems import inlet_velocity

METADATA = {**FEM_METADATA, 'cylinder_schema_version': 1, 'case': 'dfg2d2',
            'boundary': 'fixed_inlet_no_slip_walls_cylinder_natural_outlet',
            'pressure_constraint': 'natural_outflow', 'geometry': 'dfg2d2_circle_polygon',
            'lx': LENGTH, 'ly': HEIGHT, 'center_x': float(CENTER[0]), 'center_y': float(CENTER[1]),
            'radius': RADIUS, 'density': 1.,
            'physical_pressure_semantics': 'instantaneous_semidiscrete_natural_outflow'}


def representation_metadata(outlet_treatment: str = 'natural') -> dict:
    if outlet_treatment == 'natural':
        return dict(METADATA)
    if outlet_treatment != 'backflow_damped':
        raise ValueError('Unknown cylinder outlet_treatment')
    return {**METADATA, 'case': 'dfg2d2_backflow_damped',
            'boundary': 'fixed_inlet_no_slip_walls_cylinder_backflow_damped_outlet',
            'outlet_treatment': 'backflow_damped',
            'physical_pressure_semantics': 'instantaneous_semidiscrete_backflow_damped_outflow'}


def save(data: Any, model: OpenFlowModel, cfg: dict, observations: Observations, result: Result) -> None:
    for key, value in representation_metadata(cfg.get('outlet_treatment', 'natural')).items():
        data.attrs[key] = value
    data.attrs['scikit_fem_version'] = version('scikit-fem')
    expected = maps(model.velocity_basis, model.pressure_basis)
    expected['dofs/boundary'], expected['dofs/free'] = model.boundary_dofs, model.free_dofs
    expected['dofs/dirichlet_values'] = model.full_velocity(model.zero_velocity())
    for key, value in expected.items():
        data.create_dataset(key, data=value)
    assert model.velocity_basis.mesh.boundaries is not None
    for key, facets in model.velocity_basis.mesh.boundaries.items():
        data.create_dataset('mesh/boundaries/'+key, data=facets)
    data.create_dataset('pressure_probe_points', data=PROBES)
    for key in FIELDS:
        data.create_dataset('observations/'+key, data=[row[key] for row in observations.rows], dtype=float)
    available = observations.pressure_time == result.final.t and observations.pressure is not None
    data.attrs['physical_pressure_available'] = available
    if available:
        data.create_dataset('final/physical_pressure', data=observations.pressure)
        data.create_dataset('final/physical_pressure_time', data=float(result.final.t))


def validate(data: Any, cfg: dict, manifest: dict) -> tuple[Any, Any]:
    try:
        return _validate(data, cfg, manifest)
    except (KeyError, IndexError, TypeError, RuntimeError) as error:
        raise ValueError(f'Invalid cylinder reconstruction metadata: {error}') from error


def _validate(data: Any, cfg: dict, manifest: dict) -> tuple[Any, Any]:
    if data.attrs.get('schema_version') != 1 or data.attrs.get('status') != manifest['status']:
        raise ValueError('Cylinder schema/status mismatch')
    for key, value in representation_metadata(cfg.get('outlet_treatment', 'natural')).items():
        if cfg.get(key) != value or data.attrs.get(key) != value:
            raise ValueError(f'Cylinder representation mismatch: {key}')
    if data.attrs.get('pressure_semantics') != cfg['pressure_semantics']:
        raise ValueError('Cylinder stage-pressure semantics mismatch')
    if data.attrs.get('scikit_fem_version') != version('scikit-fem') \
            or manifest['source']['packages'].get('scikit-fem') != version('scikit-fem'):
        raise ValueError('Incompatible cylinder finite-element library')
    coordinates, cells = np.asarray(data['mesh/coordinates'][:]), np.asarray(data['mesh/connectivity'][:])
    if coordinates.ndim != 2 or coordinates.shape[0] != 2 or coordinates.dtype.kind != 'f' \
            or not np.isfinite(coordinates).all() or cells.ndim != 2 or cells.shape[0] != 3 \
            or cells.dtype.kind not in 'iu' or not cells.size or np.min(cells) < 0 \
            or np.max(cells) >= coordinates.shape[1]:
        raise ValueError('Invalid cylinder saved mesh arrays')
    mesh = validate_cylinder_mesh(MeshTri(coordinates, cells, sort_t=False), cfg['circle_segments'])
    velocity, pressure = MeshSpace(mesh, cfg['quadrature_order']).bases()
    assert mesh.boundaries is not None
    fixed_facets = np.concatenate([mesh.boundaries[k] for k in ('inlet', 'walls', 'cylinder')])
    fixed = velocity.get_dofs(facets=fixed_facets).all()
    expected = maps(velocity, pressure)
    expected['dofs/boundary'], expected['dofs/free'] = fixed, velocity.complement_dofs(fixed)
    trace = np.zeros(velocity.N)
    component_u = velocity.split_indices()[0]
    u, _ = inlet_velocity(velocity.doflocs[0], velocity.doflocs[1], cfg['inlet_max'])
    trace[component_u] = u[component_u]
    expected['dofs/dirichlet_values'] = trace
    for key, values in expected.items():
        saved = _array(data, key, values.shape, integer=values.dtype.kind in 'iu')
        if not np.array_equal(saved, values):
            raise ValueError(f'Cylinder saved degree-of-freedom mismatch: {key}')
    for key, facets in mesh.boundaries.items():
        if not np.array_equal(_array(data, 'mesh/boundaries/'+key, facets.shape, integer=True), facets):
            raise ValueError('Cylinder boundary tags mismatch')
    if not np.array_equal(_array(data, 'pressure_probe_points', PROBES.shape), PROBES):
        raise ValueError('Cylinder pressure-probe coordinates mismatch')
    validate_fields(data, cfg, manifest, velocity, pressure, dirichlet_dofs=fixed,
                    boundary_values=trace, pressure_gauge=False)
    times = data['times'][:]
    for key in ('boundary_flux', 'convective_boundary_power'):
        _array(data, 'diagnostics/'+key, times.shape)
    if cfg.get('outlet_treatment') == 'backflow_damped':
        damping = _array(data, 'diagnostics/outlet_backflow_dissipation', times.shape)
        if np.any(damping < 0.):
            raise ValueError('Negative outlet backflow dissipation')
    observed_times = np.asarray(data['observations/time'][:])
    count = len(observed_times)
    if observed_times.ndim != 1 or not 1 <= count <= len(times) \
            or not np.array_equal(observed_times, times[:count]):
        raise ValueError('Cylinder observations are not an accepted-state prefix')
    for key in FIELDS:
        _array(data, 'observations/'+key, (count,))
    if not np.allclose(data['observations/pressure_difference'][:],
                       data['observations/pressure_front'][:]-data['observations/pressure_rear'][:],
                       rtol=1e-12, atol=1e-13):
        raise ValueError('Cylinder pressure-difference identity mismatch')
    factor = 2/((2*cfg['inlet_max']/3)**2*(2*RADIUS))
    for coefficient, force in (('drag', 'force_x'), ('lift', 'force_y')):
        if not np.allclose(data['observations/'+coefficient][:], factor*data['observations/'+force][:],
                           rtol=1e-12, atol=1e-13):
            raise ValueError('Cylinder force normalization mismatch')
    complete_observation = count == len(times)
    available = data.attrs.get('physical_pressure_available')
    if not isinstance(available, (bool, np.bool_)) or bool(available) != complete_observation:
        raise ValueError('Cylinder physical-pressure availability mismatch')
    if complete_observation:
        _array(data, 'final/physical_pressure', (pressure.N,))
        if _array(data, 'final/physical_pressure_time', ())[()] != times[-1]:
            raise ValueError('Cylinder physical pressure has wrong time')
    elif manifest['status'] != 'failed' or 'observer failed' not in manifest.get('error', '') \
            or 'final/physical_pressure' in data:
        raise ValueError('Incomplete cylinder pressure/force observations')
    if manifest['status'] == 'complete' and not complete_observation:
        raise ValueError('Complete cylinder run lacks final observations')
    return velocity, pressure

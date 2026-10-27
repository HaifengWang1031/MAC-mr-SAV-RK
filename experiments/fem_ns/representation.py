"""Versioned FEM reconstruction metadata; never constructs a dynamical model."""
from importlib.metadata import version
from typing import Any

import h5py
import numpy as np
from skfem import Basis, ElementTriP1, ElementTriP2, ElementVector, MeshTri

from solver.fem import Space
from solver.schedule import schedule_nodes, nearest_node_indices
from solver.schemes.registry import scheme_info

METADATA = {'fem_schema_version': 1, 'boundary': 'homogeneous_no_slip',
            'discretization': 'taylor_hood_triangular', 'field_layout': 'full_component_P2_coefficients',
            'velocity_element': 'ElementVector(ElementTriP2)', 'pressure_element': 'ElementTriP1',
            'velocity_order': 2, 'pressure_order': 1,
            'pressure_constraint': 'physical_integral_zero'}
DIAGNOSTICS = ('kinetic', 'modified_energy', 'divergence_inf', 'divergence_l2',
               'continuity_residual', 'h1_seminorm_squared', 'r')


def maps(velocity: Any, pressure: Any) -> dict[str, np.ndarray]:
    """Library ordering, including edge DOFs and constraints, is part of the format."""
    components = velocity.split_indices()
    boundary = velocity.get_dofs().all()
    return {'mesh/coordinates': velocity.mesh.p, 'mesh/connectivity': velocity.mesh.t,
            'dofs/velocity_locations': velocity.doflocs, 'dofs/pressure_locations': pressure.doflocs,
            'dofs/velocity_elements': velocity.element_dofs, 'dofs/pressure_elements': pressure.element_dofs,
            'dofs/component_u': components[0], 'dofs/component_v': components[1],
            'dofs/boundary': boundary, 'dofs/free': velocity.complement_dofs(boundary)}


def save_representation(data: Any, velocity: Any, pressure: Any) -> None:
    for key, value in METADATA.items():
        data.attrs[key] = value
    data.attrs['scikit_fem_version'] = version('scikit-fem')
    for key, values in maps(velocity, pressure).items():
        data.create_dataset(key, data=values)


def _array(data: Any, key: str, shape: tuple[int, ...], *, integer: bool = False,
           finite: bool = True) -> np.ndarray:
    if key not in data:
        raise ValueError(f'Missing FEM reconstruction data: {key}')
    result = np.asarray(data[key][()])
    if result.shape != shape or result.dtype.kind not in ('iu' if integer else 'f') \
            or (finite and not np.isfinite(result).all()):
        raise ValueError(f'Invalid FEM data: {key}')
    return result


def validate(data: Any, cfg: dict, manifest: dict) -> tuple[Any, Any]:
    """Validate the entire saved representation before exposing any coefficients."""
    try:
        return _validate(data, cfg, manifest)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f'Missing or malformed FEM metadata: {error}') from error


def _validate(data: Any, cfg: dict, manifest: dict) -> tuple[Any, Any]:
    if data.attrs.get('schema_version') != 1 or data.attrs.get('status') != manifest['status']:
        raise ValueError('FEM schema/status mismatch')
    for key, value in METADATA.items():
        if data.attrs.get(key) != value or cfg.get(key) != value:
            raise ValueError(f'FEM representation mismatch: {key}')
    library = data.attrs.get('scikit_fem_version')
    if library != version('scikit-fem') or library != manifest['source']['packages']['scikit-fem']:
        raise ValueError('Incompatible FEM library version')
    info = scheme_info(cfg['scheme'])
    if data.attrs.get('pressure_semantics') != info.pressure_semantics \
            or cfg.get('pressure_semantics') != info.pressure_semantics:
        raise ValueError('FEM pressure semantics mismatch')
    expected_v, expected_p = Space(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'], cfg['quadrature_order']).bases()
    for key, expected in maps(expected_v, expected_p).items():
        saved = _array(data, key, expected.shape, integer=expected.dtype.kind in 'iu')
        if not np.array_equal(saved, expected):
            raise ValueError(f'FEM reconstruction map mismatch: {key}')
    # Rebuild the saved mesh, not a dynamical model, after checking its declared identity.
    mesh = MeshTri(data['mesh/coordinates'][:], data['mesh/connectivity'][:], sort_t=False)
    velocity = Basis(mesh, ElementVector(ElementTriP2()), intorder=cfg['quadrature_order'])
    pressure = Basis(mesh, ElementTriP1(), intorder=cfg['quadrature_order'])
    times = np.asarray(data['times'][:])
    if times.ndim != 1 or not len(times) or times.dtype.kind != 'f' or not np.isfinite(times).all() \
            or times[0] != 0. or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid FEM actual times')
    n, stages = len(times)-1, info.n_stages
    schedule = np.asarray(cfg['actual_steps'])
    if n > len(schedule) or not np.allclose(np.diff(times), schedule[:n], rtol=1e-12, atol=1e-14) \
            or (manifest['status'] == 'complete' and (n != len(schedule) or not np.isclose(times[-1], cfg['T'], rtol=1e-12, atol=1e-14))):
        raise ValueError('FEM accepted prefix/schedule mismatch')
    if not np.array_equal(_array(data, 'dt', (n,)), np.diff(times)) \
            or _array(data, 'final/t', ())[()] != times[-1]:
        raise ValueError('FEM final time mismatch')
    final_r = _array(data, 'final/r', ())[()]
    for name in DIAGNOSTICS:
        values = _array(data, 'diagnostics/'+name, (n+1,))
        if name == 'r' and (values[-1] != final_r or values[0] != cfg['r0']):
            raise ValueError('FEM scalar diagnostic mismatch')
    components = velocity.split_indices()
    boundary = velocity.get_dofs().all()

    def check_velocity(prefix: str, leading: tuple[int, ...] = ()) -> None:
        full = np.zeros((*leading, velocity.N))
        for name, indices in zip(('u', 'v'), components):
            full[..., indices] = _array(data, prefix+'/'+name, (*leading, len(indices)))
        if np.any(full[..., boundary] != 0.):
            raise ValueError('Nonzero FEM boundary coefficients')

    check_velocity('final')
    if n:
        coefficients = _array(data, 'final/stage_pressure', (stages, pressure.N))
        for p in coefficients:
            integral = float(np.sum(pressure.interpolate(p)*pressure.dx))
            if abs(integral) > 1e-8*max(1., cfg['lx']*cfg['ly']*float(np.max(np.abs(p)))):
                raise ValueError('FEM pressure integral constraint mismatch')
    elif 'final/stage_pressure' in data:
        raise ValueError('Stage pressure without accepted FEM step')
    requests = np.asarray(data['snapshots/requested_times'][:])
    if requests.ndim != 1 or not np.isfinite(requests).all():
        raise ValueError('Invalid FEM snapshot requests')
    actual = _array(data, 'snapshots/actual_times', requests.shape)
    nodes = schedule_nodes(0., schedule)
    expected_requests, expected_times = [], []
    for request, node_index in zip(cfg['snapshots'], nearest_node_indices(nodes, cfg['snapshots'])):
        if node_index <= n:
            expected_requests.append(request)
            expected_times.append(times[node_index])
    if not np.array_equal(requests, expected_requests) or not np.array_equal(actual, expected_times):
        raise ValueError('FEM requested/actual snapshot mismatch')
    if len(requests):
        check_velocity('snapshots', (len(requests),))
        _array(data, 'snapshots/r', (len(requests),))
    elif any('snapshots/'+key in data for key in ('u', 'v', 'r')):
        raise ValueError('FEM snapshot fields without times')
    for key in ('residual', 'scalar_residual', 'divergence_inf', 'continuity_residual'):
        values = _array(data, 'stages/'+key, (n, stages))
        if np.any(values < 0.):
            raise ValueError('Negative FEM stage residual')
    counts = _array(data, 'roots/count', (n, stages), integer=True)
    selected = _array(data, 'roots/selected', (n, stages))
    candidates = _array(data, 'roots/candidates', (n, stages, info.n_roots), finite=False)
    residuals = _array(data, 'roots/residuals', candidates.shape, finite=False)
    sav = cfg['scheme'].endswith('_mrsav')
    if np.any(counts < (1 if sav else 0)) or np.any(counts > info.n_roots) or (not sav and np.any(counts)):
        raise ValueError('Invalid FEM root counts')
    for index in np.ndindex(counts.shape):
        count = counts[index]
        for values in (candidates[index], residuals[index]):
            if not np.isfinite(values[:count]).all() or not np.isnan(values[count:]).all():
                raise ValueError('Invalid FEM root candidates/padding')
        if np.any(residuals[index][:count] < 0.) or (count and selected[index] not in candidates[index][:count]):
            raise ValueError('Invalid FEM selected root/residual')
    if n and selected[-1, -1] != final_r:
        raise ValueError('FEM final scalar/root mismatch')
    return velocity, pressure

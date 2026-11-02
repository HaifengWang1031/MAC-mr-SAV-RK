"""Short restarted sensitivity calculations; these do not certify the entire wake trajectory."""
import json
from pathlib import Path
from time import process_time
from typing import Any

import h5py
import numpy as np

from experiments.cylinder_ns.run import load_cylinder
from experiments.cylinder_ns.representation import validate
from experiments.cylinder_ns.reconstruct import decode_velocity
from experiments.cylinder_ns.problems import inlet_velocity
from experiments.periodic_ns.run import make_controller
from experiments.workflow import digest, write_json
from solver.adaptivity import ScalarErrorControl, integrate_adaptive
from solver.core import State
from solver.fem import CylinderSpace, MeshSpace, OpenFlowModel
from solver.schemes.registry import make_scheme


def _model(space: Any, cfg: dict) -> OpenFlowModel:
    assert space.mesh.boundaries is not None
    fixed = np.concatenate([space.mesh.boundaries[key] for key in ('inlet', 'walls', 'cylinder')])
    return OpenFlowModel(space, cfg['nu'], dirichlet_facets=fixed,
        boundary_velocity=lambda x, y: inlet_velocity(x, y, cfg['inlet_max']),
        tolerance=cfg['stokes_tolerance'], stokes_backend='recycled', cache_size=2,
        outlet_treatment=cfg.get('outlet_treatment', 'natural'))


def _sample(basis: Any, coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bound the library element-finder's Cartesian candidate array on large meshes."""
    interpolate = basis.interpolator(coefficients)
    return np.column_stack([interpolate(points[:, start:start+128])
                            for start in range(0, points.shape[1], 128)])


def _advance(model: OpenFlowModel, initial: State, cfg: dict, duration: float, *, tight: bool = False) -> State:
    controller = {**cfg['controller']}
    scalar = cfg['scalar_atol']
    if tight:
        for key in ('atol', 'rtol', 'max_step'): controller[key] *= .5
        if scalar is not None: scalar *= .5
    result = integrate_adaptive(model, make_scheme(cfg['scheme'], cfg['gamma']), initial, duration,
        min(cfg['initial_step'], controller['max_step']),
        controller=make_controller(controller, cfg['scheme']),
        scalar_control=None if scalar is None else ScalarErrorControl(scalar, cfg['gamma']))
    if result.status != 'complete':
        raise RuntimeError('Short sensitivity run failed: '+result.error)
    return result.final


def verify(directory: Path) -> Path:
    """Save sample arrays so readonly analysis can independently recompute every norm."""
    parent = json.loads((directory/'manifest.json').read_text())
    controls = json.loads((directory/'config.json').read_text())
    arrays: dict[str, Any] = {}
    rows = []
    for index, case in enumerate(parent['cases']):
        saved = json.loads((Path(case['directory'])/'result.json').read_text())
        path = Path(saved['run'])
        cfg, manifest = load_cylinder(path, require_complete=True)
        with h5py.File(path/'results.h5', 'r') as data:
            basis, _ = validate(data, cfg, manifest)
            initial = State(float(data['final/t'][()]), data['final/u'][:], data['final/v'][:], float(data['final/r'][()]))
        start = process_time()
        model = _model(MeshSpace(basis.mesh, cfg['quadrature_order']), cfg)
        full = decode_velocity(basis, initial.u, initial.v)
        base_initial = model.state(initial.t, full[model.free_dofs], initial.r)
        base = _advance(model, base_initial, cfg, controls['check_duration'])
        tight = _advance(model, base_initial, cfg, controls['check_duration'], tight=True)
        fine = _model(CylinderSpace(cfg['resolution']*controls['check_resolution_factor'],
                                  cfg['circle_segments'], cfg['quadrature_order']), cfg)
        # Dirichlet values are known exactly; do not locate shared polygon corners
        # using a floating-point element finder. Sample the remaining nodes only.
        free_nodes = ~np.isin(fine.component_dofs[0], fine.boundary_dofs)
        query = fine.velocity_basis.doflocs[:, fine.component_dofs[0][free_nodes]].copy()
        outlet_shift = 64*np.finfo(float).eps*cfg['lx']
        query[0] = np.minimum(query[0], cfg['lx']-outlet_shift)
        sampled = _sample(basis, full, query)
        fine_full = fine.full_velocity(fine.zero_velocity())
        for values, indices in zip(sampled, fine.component_dofs): fine_full[indices[free_nodes]] = values
        projected = fine.solve(fine_full[fine.free_dofs], mass=1., viscosity=0.)
        fine_initial = fine.state(initial.t, projected.velocity, initial.r)
        final = _advance(fine, fine_initial, cfg, controls['check_duration'])
        points = basis.global_coordinates().reshape(2, -1)

        def samples(owner: OpenFlowModel, state: State) -> np.ndarray:
            values = decode_velocity(owner.velocity_basis, state.u, state.v)
            return _sample(owner.velocity_basis, values, points).reshape(2, *basis.dx.shape)

        before, mapped = samples(model, base_initial), samples(fine, fine_initial)
        a, b, c = samples(model, base), samples(model, tight), samples(fine, final)
        def relative(left: np.ndarray, right: np.ndarray) -> float:
            return float(np.sqrt(np.sum((left-right)**2*basis.dx)/np.sum(right**2*basis.dx)))
        time_error, space_error, transfer_error = relative(a, b), relative(a, c), relative(before, mapped)
        prefix = f'case_{index}_'
        for key, values in (('weights', basis.dx), ('base', a), ('tight', b), ('fine', c),
                            ('initial', before), ('mapped_initial', mapped)):
            arrays[prefix+key] = values
        rows.append({'reynolds': cfg['reynolds'], 'run': str(path), 'results_sha256': manifest['results_sha256'],
            'time_relative_L2': time_error, 'space_relative_L2': space_error, 'initial_transfer_relative_L2': transfer_error,
            'base_triangles': int(basis.mesh.t.shape[1]), 'fine_triangles': int(fine.velocity_basis.mesh.t.shape[1]),
            'outlet_probe_inward_shift': outlet_shift,
            'start_time': initial.t, 'end_time': base.t, 'cpu_seconds': process_time()-start,
            'time_passed': time_error <= controls['short_time_gate'],
            'space_passed': space_error <= controls['short_space_gate']})
    with (directory/'checks.npz').open('wb') as stream: np.savez_compressed(stream, **arrays)
    result = {'status': 'complete', 'rows': rows, 'samples_sha256': digest(directory/'checks.npz'),
        'duration': controls['check_duration'], 'time_gate': controls['short_time_gate'],
        'space_gate': controls['short_space_gate'],
        'limits': 'Short restarted sensitivity only; space comparison includes initial interpolation and weak projection. '
                  'Not full-trajectory convergence, turbulence validation, or a benchmark error bound.'}
    write_json(directory/'checks.json', result)
    return directory/'checks.json'

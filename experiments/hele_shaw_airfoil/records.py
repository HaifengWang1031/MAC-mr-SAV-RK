"""Read actual saved meshes and fields; never generates a mesh or advances a flow."""
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from skfem import MeshTri

from experiments.fem_ns.representation import maps, validate_fields
from experiments.records import load_record, run_identity as identity
from solver.fem.airfoil import airfoil_polygon, validate_airfoil_mesh
from solver.fem.space import MeshSpace
from solver.integrate import step_sizes
from solver.schemes.registry import scheme_info
from .config import DEFAULTS, OPTIONAL, METADATA, effective_config


def load(directory: Path, *, require_complete: bool = True) -> tuple[dict, dict, MeshTri, Any, Any]:
    """Validate checksums, config, DOF maps, all stages/roots and physical pressure."""
    directory = Path(directory)
    cfg, manifest = load_record(directory, require_complete=require_complete)
    if manifest.get('experiment') != 'hele_shaw_airfoil' or manifest.get('run_id') != directory.name:
        raise ValueError('Expected Hele-Shaw airfoil identity')
    inputs = {k: cfg[k] for k in DEFAULTS}
    inputs.update({k: cfg[k] for k in OPTIONAL if k in cfg})
    fresh = effective_config(inputs)
    legacy = cfg.get('hele_shaw_schema_version', 0) == 0
    if legacy:
        # Initial development records inherited the registry's mean_zero text,
        # although the saved constraint and actual model used pressure tractions.
        # Preserve their exact raw metadata and identity, with an explicit adapter.
        fresh.pop('hele_shaw_schema_version')
        fresh['pressure_semantics'] = scheme_info(cfg['scheme']).pressure_semantics+'_normalized_MWR_pressure_traction'
        if cfg['mode'] == 'fixed':
            fresh['actual_steps'] = step_sizes(cfg['T'], dt=cfg['dt']).tolist()
    if fresh != cfg \
            or identity(cfg, manifest['source']) != manifest['identity']:
        raise ValueError('Hele-Shaw configuration/source identity mismatch')
    with h5py.File(directory/'results.h5', 'r') as data:
        if data.attrs.get('scikit_fem_version') != version('scikit-fem') \
                or manifest['source']['packages']['scikit-fem'] != version('scikit-fem'):
            raise ValueError('Incompatible finite-element library version')
        if data.attrs.get('status') != manifest['status'] or data.attrs.get('pressure_semantics') != cfg['pressure_semantics']:
            raise ValueError('Status/pressure representation mismatch')
        if any(data.attrs.get(k) != (json.dumps(cfg[k]) if isinstance(cfg[k], list) else cfg[k])
               for k in METADATA if not (legacy and k == 'hele_shaw_schema_version')):
            raise ValueError('Model/geometry representation mismatch')
        if 'inlet_treatment' in cfg and data.attrs.get('inlet_treatment') != cfg['inlet_treatment']:
            raise ValueError('Inlet boundary representation mismatch')
        mesh = MeshTri(data['mesh/coordinates'][:], data['mesh/connectivity'][:], sort_t=False)
        mesh = mesh.with_boundaries({k: data['boundaries/'+k][:] for k in ('inlet', 'outlet', 'walls', 'airfoil')})
        validate_airfoil_mesh(mesh, airfoil_polygon(cfg['chord'], cfg['angle'], cfg['surface_points']),
                              tuple(np.asarray(cfg['bounds'])*cfg['chord']))
        vb, pb = MeshSpace(mesh, cfg['quadrature_order']).bases()
        fixed = vb.get_dofs(facets=np.concatenate([mesh.boundaries[k] for k in ('walls', 'airfoil')])).all()
        expected = maps(vb, pb)
        expected['dofs/boundary'], expected['dofs/free'] = fixed, vb.complement_dofs(fixed)
        for key, values in expected.items():
            if not np.array_equal(data[key][:], values):
                raise ValueError(f'Invalid saved FE map: {key}')
        validate_fields(data, cfg, manifest, vb, pb, dirichlet_dofs=fixed, pressure_gauge=False)
        times = data['snapshots/actual_times'][:]
        p = data['snapshots/physical_pressure'][:]
        if p.shape != (len(times), pb.N) or not np.isfinite(p).all() \
                or data.attrs.get('snapshot_pressure_semantics') != 'instantaneous_physical_MWR_model_pressure':
            raise ValueError('Invalid physical snapshot pressure')
        force = data['snapshots/model_force_per_span'][:]
        if force.shape != (len(times), 2) or not np.isfinite(force).all():
            raise ValueError('Invalid model-force observations')
        if cfg['darcy_reference']:
            scalar_size = len(vb.split_indices()[0])
            for name in ('pressure', 'u', 'v'):
                values = data['darcy/'+name][:]
                if values.shape != (scalar_size,) or not np.isfinite(values).all():
                    raise ValueError('Invalid Darcy reference fields')
    return cfg, manifest, mesh, vb, pb

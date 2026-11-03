"""Validate explicit native-state refinement/benchmark records without advancing a model."""
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from skfem import BilinearForm, MeshTri, asm
from skfem.helpers import div

import h5py
from experiments.records import digest, load_record, run_identity as _identity
from .config import DEFAULTS, OPTIONAL, effective_config
from .representation import validate
from solver.fem.cylinder import validate_cylinder_mesh
from solver.fem.space import MeshSpace
from .problems import inlet_velocity
from .representation import decode_velocity


def fingerprints(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): digest(p) for p in directory.rglob('*') if p.is_file()}


def load_jobs(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify artifact identities and actual duration, including nonzero-time restarts."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['run_id'] != directory.name or manifest['status'] != 'complete' or manifest['experiment'] not in (
            'fem_adaptive_benchmark', 'cylinder_temporal_reference', 'cylinder_refinement_reference'):
        raise ValueError('Expected an explicitly completed native-state campaign')
    if set(manifest['input_sha256']) != {'config.json', 'native.npz'}:
        raise ValueError('Incomplete native input identity')
    for name, checksum in manifest['input_sha256'].items():
        if digest(directory/'input'/name) != checksum:
            raise ValueError('Native input checksum mismatch')
    jobs = []
    for job in manifest['jobs']:
        if job['label'] not in ('baseline', 'direct', 'recycled') or job['mode'] not in ('adaptive', 'fixed', 'replay', 'profile') \
                or type(job['repeat']) is not int or job['repeat'] < 0:
            raise ValueError('Invalid native worker identity')
        prefix = directory/f"{job['mode']}-{job['label']}-{job['repeat']}"
        required = {'.json', '.npz', '.log'} | ({'.pstats'} if job['profiled'] else set())
        if set(job['artifacts']) != required:
            raise ValueError('Incomplete worker artifact identity')
        for suffix, checksum in job['artifacts'].items():
            if digest(prefix.with_suffix(suffix)) != checksum:
                raise ValueError('Worker artifact checksum mismatch')
        result = json.loads(prefix.with_suffix('.json').read_text())
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as data:
            fields = {key: data[key] for key in data.files}
        times = fields['times']
        if result['status'] != 'complete' or len(times) < 2 or not np.isfinite(times).all() \
                or np.any(np.diff(times) <= 0) or times[0] != result['initial_time'] or times[-1] != result['final_time'] \
                or not np.isclose(times[-1]-times[0], manifest['duration'], rtol=0., atol=1e-8) \
                or result['duration'] != manifest['duration']:
            raise ValueError('Invalid completed worker duration/accepted nodes')
        for key in ('observation_time', 'drag', 'lift', 'pressure'):
            if fields[key].shape != times.shape or not np.isfinite(fields[key]).all():
                raise ValueError('Incomplete physical observation history')
        if not np.array_equal(fields['observation_time'], times):
            raise ValueError('Observations are not at accepted states')
        accepted = fields['accepted']
        if accepted.dtype.kind != 'b' or fields['attempt_h'].shape != accepted.shape \
                or np.any(~np.isfinite(fields['attempt_h'])) or np.any(fields['attempt_h'] <= 0.) \
                or int(np.sum(accepted)) != len(times)-1:
            raise ValueError('Invalid native attempted-step history')
        checks = [result['metrics'][key] for key in ('max_stage_residual', 'max_continuity')]
        if not manifest['duration'] > 0. or not np.isfinite(checks).all() or min(checks) < 0. \
                or checks[0] > 1e-8 or not np.isfinite(fields['r']).all() \
                or result['source']['packages']['scikit-fem'] != version('scikit-fem'):
            raise ValueError('Worker did not satisfy stage/library checks')
        jobs.append({'job': job, 'result': result, 'fields': fields, 'prefix': prefix})
    if not jobs:
        raise ValueError('No completed workers')
    return manifest, jobs


def load_reference(directory: Path) -> dict[str, Any]:
    """Read saved native mesh, final/snapshot P2 fields and physical measurements only."""
    directory = Path(directory)
    manifest, jobs = load_jobs(directory)
    if manifest['experiment'] == 'fem_adaptive_benchmark' or len(jobs) != 1 \
            or jobs[0]['job']['mode'] not in ('fixed', 'adaptive'):
        raise ValueError('Expected one fixed/adaptive reference worker')
    cfg = json.loads((directory/'input/config.json').read_text())
    with np.load(directory/'input/native.npz', allow_pickle=False) as data:
        native = {key: data[key] for key in data.files}
    mesh = MeshTri(native['coordinates'], native['connectivity'], sort_t=False)
    mesh = validate_cylinder_mesh(mesh, cfg['circle_segments'])
    assert mesh.boundaries is not None
    for key in ('inlet', 'outlet', 'walls', 'cylinder'):
        if not np.array_equal(native['boundary_'+key], mesh.boundaries[key]):
            raise ValueError('Saved refinement boundary tags disagree with geometry')
    vb, pb = MeshSpace(mesh, cfg['quadrature_order']).bases()
    parts = vb.split_indices()
    if any(not np.array_equal(indices, native['component_'+name]) for name, indices in zip(('u', 'v'), parts)):
        raise ValueError('Saved refinement P2 component maps disagree')
    item = jobs[0]
    fields, result = item['fields'], item['result']
    if result['step'] != manifest['fixed_step'] or result['metrics']['max_continuity'] > cfg['stokes_tolerance']:
        raise ValueError('Invalid reference step/complete weak-continuity check')
    if 'scalar_history' in fields and (fields['scalar_history'].shape != fields['times'].shape \
            or not np.isfinite(fields['scalar_history']).all() or fields['scalar_history'][-1] != fields['r']):
        raise ValueError('Invalid reference scalar history')
    if 'physical_pressure' in fields and (fields['physical_pressure'].shape != (pb.N,) \
            or not np.isfinite(fields['physical_pressure']).all()):
        raise ValueError('Invalid final physical P1 pressure coefficients')
    fixed = vb.get_dofs(facets=np.concatenate([mesh.boundaries[key] for key in ('inlet', 'walls', 'cylinder')])).all()
    expected = np.zeros(vb.N)
    for component, indices in enumerate(parts):
        expected[indices] = inlet_velocity(vb.doflocs[0, indices], vb.doflocs[1, indices], cfg['inlet_max'])[component]
    for name in ('u', 'v'):
        if fields[name].shape != (len(parts[0]),) or not np.isfinite(fields[name]).all():
            raise ValueError('Invalid reference final P2 coefficients')
    full = decode_velocity(vb, fields['u'], fields['v'])
    if np.max(abs(full[fixed]-expected[fixed])) > 1e-12:
        raise ValueError('Reference final velocity violates its saved fixed trace')
    @BilinearForm
    def continuity(velocity: Any, pressure: Any, w: Any) -> Any:
        return div(velocity)*pressure
    if np.max(abs(asm(continuity, vb, pb)@full)) > cfg['stokes_tolerance']:
        raise ValueError('Reference final field violates a full pressure-test continuity row')
    if fields['snapshot_u'].shape != fields['snapshot_v'].shape \
            or fields['snapshot_u'].shape != (len(fields['snapshot_times']), len(parts[0])) \
            or not np.isfinite(fields['snapshot_u']).all() or not np.isfinite(fields['snapshot_v']).all() \
            or np.any(~np.isin(fields['snapshot_times'], fields['times'])):
        raise ValueError('Invalid reference saved-node snapshots')
    for name, indices in zip(('u', 'v'), parts):
        local = np.flatnonzero(np.isin(indices, fixed))
        if np.max(abs(fields['snapshot_'+name][:, local]-expected[indices[local]]), initial=0.) > 1e-12:
            raise ValueError('Reference snapshot violates its fixed trace')
    return {'directory': directory, 'manifest': manifest, 'config': cfg, 'basis': vb, 'pressure_basis': pb,
            'result': result, 'fields': fields, 'native': native}


def load_cylinder(directory: Path, *, require_complete: bool = False) -> tuple[dict, dict]:
    directory = Path(directory)
    cfg, manifest = load_record(directory, require_complete=require_complete)
    if manifest.get('experiment') != 'cylinder_ns' or manifest.get('run_id') != directory.name:
        raise ValueError('Expected cylinder run identity')
    try:
        source = manifest['source']
        packages = ('numpy', 'scipy', 'numba', 'h5py', 'sympy', 'matplotlib', 'scikit-fem')
        if any(not isinstance(source.get(k), str) or not source[k] for k in ('code_sha256', 'python', 'platform')) \
                or len(source['code_sha256']) != 64 or any(c not in '0123456789abcdef' for c in source['code_sha256']) \
                or any(not isinstance(source['packages'].get(k), str) or not source['packages'][k] for k in packages) \
                or not isinstance(source['threads'], dict):
            raise ValueError('Malformed cylinder provenance')
        inputs = {key: cfg.get(key, value) for key, value in DEFAULTS.items()}
        for key in OPTIONAL:
            if key in cfg:
                inputs[key] = cfg[key]
        if effective_config(inputs) != cfg or _identity(cfg, source) != manifest['identity']:
            raise ValueError('Cylinder effective configuration/source identity mismatch')
        with h5py.File(directory/'results.h5', 'r') as data:
            validate(data, cfg, manifest)
    except (KeyError, TypeError) as error:
        raise ValueError(f'Missing cylinder metadata: {error}') from error
    return cfg, manifest

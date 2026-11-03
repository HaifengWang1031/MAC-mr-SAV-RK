"""Read and validate saved representation records without importing the runner."""
from pathlib import Path
import h5py
from experiments.records import load_record
from .config import DEFAULTS, OPTIONS, effective_config
from .representation import validate
from experiments.records import run_identity as _identity


def load_fem(directory: Path, *, require_complete: bool = False) -> tuple[dict, dict]:
    """Integrity and representation gate, also used by every readonly consumer."""
    directory = Path(directory)
    cfg, manifest = load_record(directory, require_complete=require_complete)
    if manifest.get('experiment') != 'fem_ns' or manifest.get('run_id') != directory.name:
        raise ValueError('Expected FEM run identity')
    source = manifest.get('source', {})
    packages = ('numpy', 'scipy', 'numba', 'h5py', 'sympy', 'matplotlib', 'scikit-fem')
    threads = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')
    if not isinstance(source, dict) or any(not isinstance(source.get(key), str) or not source[key]
                                            for key in ('code_sha256', 'python', 'platform')) \
            or len(source['code_sha256']) != 64 \
            or any(c not in '0123456789abcdef' for c in source['code_sha256']) \
            or not isinstance(source.get('packages'), dict) \
            or any(not isinstance(source['packages'].get(key), str) or not source['packages'][key] for key in packages) \
            or not isinstance(source.get('threads'), dict) \
            or any(key not in source['threads'] or (source['threads'][key] is not None
                   and not isinstance(source['threads'][key], str)) for key in threads):
        raise ValueError('Missing or malformed FEM provenance')
    try:
        inputs = {key: cfg[key] for key in DEFAULTS if key != 'dt' or 'steps' not in cfg}
        if 'steps' in cfg:
            inputs['steps'] = cfg['steps']
        inputs.update({key: cfg[key] for key in OPTIONS if key in cfg})
        if effective_config(inputs) != cfg or _identity(cfg, manifest['source']) != manifest['identity']:
            raise ValueError('FEM effective config/source identity mismatch')
        with h5py.File(directory/'results.h5', 'r') as data:
            validate(data, cfg, manifest)
    except (KeyError, TypeError) as error:
        raise ValueError(f'Missing FEM identity/metadata: {error}') from error
    return cfg, manifest

"""Read and validate saved representation records without importing the runner."""
from pathlib import Path
import h5py
import numpy as np
from experiments.records import load_record
from solver.fourier import FourierSpace
from solver.schemes.registry import scheme_info


def load_periodic(directory: Path, *, require_complete: bool = False) -> tuple[dict, dict]:
    cfg, manifest = load_record(directory, require_complete=require_complete)
    if manifest.get('experiment') != 'periodic_ns':
        raise ValueError('Expected a periodic_ns record')
    s = FourierSpace(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    info = scheme_info(cfg['scheme'])
    for key, value in s.metadata().items():
        if cfg.get(key) != value:
            raise ValueError(f'Periodic representation mismatch: {key}')
    with h5py.File(directory/'results.h5', 'r') as data:
        if data.attrs.get('status') != manifest['status']:
            raise ValueError('Periodic status mismatch')
        for key in ('boundary', 'discretization', 'field_layout', 'fft_norm', 'dealias'):
            if data.attrs.get(key) != cfg[key]:
                raise ValueError(f'HDF5 periodic representation mismatch: {key}')
        if data.attrs.get('pressure_semantics') != cfg.get('pressure_semantics') \
                or cfg.get('pressure_semantics') != info.pressure_semantics:
            raise ValueError('Periodic pressure semantics mismatch')
        if 'final/stage_pressure' in data:
            pressure = data['final/stage_pressure'][:]
            if pressure.shape != (info.n_stages, *s.shape) or not np.isfinite(pressure).all():
                raise ValueError('Invalid periodic stage pressure')
        for component in ('u', 'v'):
            if data['final/'+component].shape != s.shape \
                    or data['final/'+component].dtype.kind != 'f':
                raise ValueError('Periodic field layout/dtype mismatch')
    return cfg, manifest

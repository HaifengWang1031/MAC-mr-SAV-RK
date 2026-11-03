"""Read and validate saved representation records without importing the runner."""
from pathlib import Path
import h5py
import numpy as np
from experiments.records import load_record
from solver.channel import ChannelGrid
from solver.schemes.registry import scheme_info


def load_channel(directory: Path, *, require_complete: bool = False) -> tuple[dict, dict]:
    cfg, manifest = load_record(directory, require_complete=require_complete)
    if manifest.get('experiment') != 'channel_ns': raise ValueError('Expected channel_ns record')
    grid = ChannelGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    if cfg.get('backend') not in ('direct', 'fft'): raise ValueError('Unsupported channel backend')
    with h5py.File(directory/'results.h5', 'r') as data:
        if data.attrs.get('status') != manifest['status']: raise ValueError('Channel status mismatch')
        for key, value in grid.metadata().items():
            if cfg.get(key) != value or data.attrs.get(key) != value:
                raise ValueError(f'Channel representation mismatch: {key}')
        if data.attrs.get('pressure_semantics') != cfg.get('pressure_semantics'):
            raise ValueError('Channel pressure semantics mismatch')
        if data['final/u'].dtype.kind != 'f' or data['final/v'].dtype.kind != 'f':
            raise ValueError('Channel fields must be real')
        grid.pack(data['final/u'][:], data['final/v'][:])
        stages = scheme_info(cfg['scheme']).n_stages
        if 'final/stage_pressure' in data:
            p = data['final/stage_pressure'][:]
            if p.shape != (stages, grid.ny, grid.nx) or not np.isfinite(p).all():
                raise ValueError('Invalid channel stage pressure')
        times = data['times'][:]
        if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) \
                or times[-1] != data['final/t'][()] or not np.isfinite(data['final/r'][()]):
            raise ValueError('Invalid channel times/final state')
        if manifest['status'] == 'complete' and not np.isclose(times[-1], cfg['T'], rtol=1e-12, atol=1e-14):
            raise ValueError('Complete channel run has wrong endpoint')
        if 'snapshots/u' in data:
            for u, v in zip(data['snapshots/u'], data['snapshots/v']): grid.pack(u, v)
    return cfg, manifest

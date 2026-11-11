"""Read and validate pressure-free periodic MHD records."""
from pathlib import Path

import h5py
import numpy as np

from run_support.records import load_record
from solver.fourier import FourierSpace
from solver.schemes.registry import scheme_info
from .zhou import ZHOU_CASE


def load_mhd(directory: Path, *, require_complete: bool = False) -> tuple[dict, dict]:
    cfg, manifest = load_record(directory, require_complete=require_complete)
    if manifest.get('experiment') != 'periodic_mhd':
        raise ValueError('Expected a periodic_mhd record')
    space = FourierSpace(cfg['nx'], cfg['ny'])
    if any(cfg.get(key) != value for key, value in space.metadata().items()):
        raise ValueError('MHD Fourier representation mismatch')
    with h5py.File(directory/'results.h5', 'r') as data:
        if data.attrs.get('status') != manifest['status'] \
                or data.attrs.get('pressure_semantics') != 'leray_eliminated' \
                or 'final/stage_pressure' in data:
            raise ValueError('Invalid projected MHD status or pressure layout')
        if data['final/u'].shape != space.shape or data['final/v'].shape != space.shape \
                or data['final/magnetic'].shape != (2, *space.shape):
            raise ValueError('Invalid MHD field shape')
        actual = data['snapshots/actual_times'][:]
        if data['snapshots/magnetic'].shape != (len(actual), 2, *space.shape):
            raise ValueError('Invalid MHD magnetic snapshots')
        if cfg['case'] == ZHOU_CASE and any(data[f'snapshots/{name}'].shape != (len(actual), *space.shape)
                                           for name in ('flux', 'current')):
            raise ValueError('Invalid Zhou magnetic geometry snapshots')
        times = data['times'][:]
        if manifest['status'] == 'complete' and (
                abs(times[-1]-cfg['T']) > 1e-12 or len(actual) != len(cfg['snapshots'])
                or not np.allclose(actual, cfg['snapshots'], rtol=0, atol=1e-12)):
            raise ValueError('Incomplete MHD time coverage')
        counts = data['corrections/count'][:]
        info = scheme_info(cfg['scheme'])
        if counts.shape != (len(times)-1, info.n_stages) or np.any(counts != info.n_corrections):
            raise ValueError('MHD stage correction count mismatch')
    return cfg, manifest

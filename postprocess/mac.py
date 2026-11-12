"""Read saved MAC face arrays and diagnostics without constructing a flow model."""
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from run_support.records import load_record
from solver.mac.grid import MACGrid, Array


def read_run_fields(path: Path) -> tuple[dict[str, Any], MACGrid, list[float],
                                         dict[str, dict[float, Array]], dict[str, Array]]:
    """Complete saved snapshots: cell-centred speed, interior-vertex curl, history.

    The historical tuple interface, strict requested/actual node rule and field
    locations are retained. This is not a Fourier, Channel or FEM reader.
    """
    cfg, _ = load_record(path, require_complete=True)
    grid = MACGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    times: list[float] = []
    fields: dict[str, dict[float, Array]] = {'speed': {}, 'vorticity': {}}
    with h5py.File(path/'results.h5') as stored:
        for metadata in (cfg, stored.attrs):
            if metadata.get('discretization', 'mac') != 'mac' \
                    or metadata.get('field_layout', 'staggered_mac') != 'staggered_mac' \
                    or metadata.get('boundary', 'homogeneous_no_slip') not in (
                        'homogeneous_no_slip', 'moving_top_lid_stationary_other_walls'):
                raise ValueError('Unsupported MAC record layout; use its dedicated reader')
        requested_times = stored['snapshots/requested_times'][:]
        actual_times = stored['snapshots/actual_times'][:]
        if requested_times.ndim != 1 or requested_times.shape != actual_times.shape \
                or not np.isfinite(requested_times).all() or not np.isfinite(actual_times).all():
            raise ValueError('Snapshot time arrays must have matching finite entries')
        for component, shape in (('u', (grid.ny, grid.nx+1)), ('v', (grid.ny+1, grid.nx))):
            key = 'snapshots/'+component
            if key in stored:
                if stored[key].shape != (len(requested_times), *shape):
                    raise ValueError('MAC snapshot shape does not match the saved grid/time count')
            elif len(requested_times):
                raise ValueError('Missing MAC snapshot component: '+component)
        for index, (requested, actual) in enumerate(zip(requested_times, actual_times, strict=True)):
            if abs(float(actual)-float(requested)) > 1e-9*max(1., abs(float(requested))):
                raise ValueError('Snapshot is not on the requested time node')
            u = stored['snapshots/u'][index]
            v = stored['snapshots/v'][index]
            if not np.isfinite(u).all() or not np.isfinite(v).all():
                raise ValueError('Nonfinite MAC snapshot field')
            centre_u = 0.5*(u[:, :-1]+u[:, 1:])
            centre_v = 0.5*(v[:-1, :]+v[1:, :])
            time = float(requested)
            times.append(time)
            fields['speed'][time] = np.hypot(centre_u, centre_v)
            fields['vorticity'][time] = ((v[1:-1, 1:]-v[1:-1, :-1])/grid.hx
                                         -(u[1:, 1:-1]-u[:-1, 1:-1])/grid.hy)
        history = {name: stored['diagnostics/'+name][:] for name in stored['diagnostics']}
        history['time'] = stored['times'][:]
    return cfg, grid, sorted(times), fields, history

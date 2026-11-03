"""Read saved P2 velocity/P1 pressures without constructing a flow model."""
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from .representation import decode_velocity, validate
from .records import load_cylinder


def reconstruct(directory: Path, points: NDArray[np.float64], *, snapshot: int | None = None) -> dict[str, Any]:
    """Snapshot pressure is absent; physical and incremental pressures are distinguished."""
    cfg, manifest = load_cylinder(directory)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] != 2 or not points.shape[1] or not np.isfinite(points).all() \
            or np.any(points < 0.) or np.any(points > np.array([[cfg['lx']], [cfg['ly']]])):
        raise ValueError('Expected finite physical points (2, n) in the saved fluid mesh')
    with h5py.File(Path(directory)/'results.h5', 'r') as data:
        vb, pb = validate(data, cfg, manifest)
        if snapshot is None:
            u, v = data['final/u'][:], data['final/v'][:]
            time, r = float(data['final/t'][()]), float(data['final/r'][()])
        else:
            count = len(data['snapshots/actual_times'])
            if type(snapshot) is not int or not 0 <= snapshot < count:
                raise ValueError('Invalid cylinder snapshot index')
            u, v = data['snapshots/u'][snapshot], data['snapshots/v'][snapshot]
            time, r = float(data['snapshots/actual_times'][snapshot]), float(data['snapshots/r'][snapshot])
        full = decode_velocity(vb, u, v)
        physical_pressure, stage_pressure = None, None
        if snapshot is None:
            if 'final/physical_pressure' in data:
                physical_pressure = np.asarray(pb.interpolator(data['final/physical_pressure'][:])(points))
            if 'final/stage_pressure' in data:
                stage_pressure = np.stack([pb.interpolator(p)(points) for p in data['final/stage_pressure'][:]])
        return {'velocity': np.asarray(vb.interpolator(full)(points)), 'physical_pressure': physical_pressure,
                'stage_pressure': stage_pressure, 'time': time, 'r': r,
                'u_coefficients': u, 'v_coefficients': v,
                'pressure_semantics': cfg['pressure_semantics'],
                'physical_pressure_semantics': cfg['physical_pressure_semantics']}

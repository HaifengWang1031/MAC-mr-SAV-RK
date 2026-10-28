"""Evaluate full saved P2/P1 fields using only a readonly mesh and bases."""
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from .representation import validate
from .run import load_fem


def reconstruct(directory: Path, points: NDArray[np.float64], *, snapshot: int | None = None) -> dict[str, Any]:
    """Return evaluated velocity and last accepted stage-pressure multipliers.

    Snapshot indices address saved snapshots, not arbitrary trajectory nodes. Snapshot
    pressure is deliberately absent: only the final accepted step's stages are saved.
    """
    cfg, manifest = load_fem(directory)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] != 2 or not points.shape[1] or not np.isfinite(points).all() \
            or np.any(points < 0.) or np.any(points > np.array([[cfg['lx']], [cfg['ly']]])):
        raise ValueError('Expected finite physical points (2, n) inside the saved rectangle')
    with h5py.File(Path(directory)/'results.h5', 'r') as data:
        vb, pb = validate(data, cfg, manifest)
        if snapshot is None:
            u, v = data['final/u'][:], data['final/v'][:]
            time, r = float(data['final/t'][()]), float(data['final/r'][()])
        else:
            count = len(data['snapshots/actual_times'])
            if type(snapshot) is not int or not 0 <= snapshot < count:
                raise ValueError('Invalid saved FEM snapshot index')
            u, v = data['snapshots/u'][snapshot], data['snapshots/v'][snapshot]
            time, r = float(data['snapshots/actual_times'][snapshot]), float(data['snapshots/r'][snapshot])
        full = np.zeros(vb.N)
        for coefficients, indices in zip((u, v), vb.split_indices()):
            full[indices] = coefficients
        pressure = None
        if snapshot is None and 'final/stage_pressure' in data:
            pressure = np.stack([pb.interpolator(p)(points) for p in data['final/stage_pressure'][:]])
        return {'velocity': np.asarray(vb.interpolator(full)(points)), 'stage_pressure': pressure,
                'time': time, 'r': r, 'u_coefficients': u, 'v_coefficients': v,
                'pressure_semantics': cfg['pressure_semantics']}

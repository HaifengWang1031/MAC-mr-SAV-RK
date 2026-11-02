"""Validated configuration for lid-driven wedge-cavity runs.

Every derived quantity that appears in a record is recomputed here from the
inputs, so a saved `config.json` can be re-validated without the run directory.
"""
from typing import Any

import numpy as np
from experiments.fem_backends import FEM_BACKEND_KEYS, validate_solver_options

from solver.fem.wedge import mesh_parameters
from solver.schemes.registry import SCHEMES

DEFAULTS: dict[str, Any] = {
    'apex_angle': 90.0,
    'lid_length': 1.0,
    'lid_cells': 64,
    'lid_speed': 1.0,
    'nu': 0.01,
    'T': 60.0,
    'dt': 0.01,
    'scheme': 'sdirk2_mrsav',
    'gamma': 1.0,
    'r0': 0.0,
    'lid_corner_policy': 'stationary',
    'snapshot_every': 0.1,
    'log_every': 500,
    'quadrature_order': 6,
    'stokes_backend': 'direct',
    'stokes_tolerance': 1e-9,
}
OPTIONAL = ('case', 'note')+FEM_BACKEND_KEYS


def _positive(cfg: dict, key: str) -> float:
    value = cfg[key]
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{key} must be finite and positive')
    return float(value)


def effective_config(config: dict) -> dict:
    """Fill defaults, validate, and append the derived geometry and schedule."""
    unknown = set(config) - set(DEFAULTS) - set(OPTIONAL)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    cfg = {**DEFAULTS, **config}
    validate_solver_options(cfg)
    if cfg['scheme'] not in SCHEMES:
        raise ValueError('Unknown scheme')
    if type(cfg['lid_cells']) is not int:
        raise ValueError('lid_cells must be an integer')
    if not np.isfinite(cfg['gamma']) or cfg['gamma'] < 0:
        raise ValueError('gamma must be finite and nonnegative')
    if not np.isfinite(cfg['r0']):
        raise ValueError('r0 must be finite')
    if cfg['lid_corner_policy'] not in ('stationary', 'lid'):
        raise ValueError('lid_corner_policy must be stationary or lid')
    if type(cfg['log_every']) is not int or cfg['log_every'] < 1:
        raise ValueError('log_every must be a positive integer')
    if type(cfg['quadrature_order']) is not int or cfg['quadrature_order'] < 5:
        raise ValueError('quadrature_order must be an integer >= 5')
    if not np.isfinite(cfg['stokes_tolerance']) or cfg['stokes_tolerance'] <= 0:
        raise ValueError('stokes_tolerance must be positive')
    for key in ('lid_speed', 'nu', 'T', 'dt'):
        _positive(cfg, key)
    alpha, half_width, height, n = mesh_parameters(cfg['lid_length'], cfg['apex_angle'], cfg['lid_cells'])
    if cfg['snapshot_every'] is not None:
        if not np.isfinite(cfg['snapshot_every']) or cfg['snapshot_every'] <= 0:
            raise ValueError('snapshot_every must be positive')
        count = int(np.floor(cfg['T']/cfg['snapshot_every'] + 1e-9))
        snapshots = [round(cfg['snapshot_every']*(index+1), 12) for index in range(count)]
    else:
        snapshots = []
    if any(t > cfg['T']+1e-12 for t in snapshots):
        raise ValueError('Snapshot time beyond T')
    steps = int(np.round(cfg['T']/cfg['dt']))
    if steps < 1 or abs(steps*cfg['dt']-cfg['T']) > 1e-9*max(1., cfg['T']):
        raise ValueError('dt must divide T into a whole number of steps')
    cfg.update(apex_half_angle=float(np.rad2deg(alpha)), half_width=half_width, height=height,
               cells_per_half_lid=n, steps=steps, snapshot_times=snapshots,
               reynolds=cfg['lid_speed']*cfg['lid_length']/cfg['nu'],
               element_size=[half_width/n, height/n], area=half_width*height)
    return cfg


DERIVED = ('apex_half_angle', 'half_width', 'height', 'cells_per_half_lid', 'steps',
           'snapshot_times', 'reynolds', 'element_size', 'area')


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        return isinstance(left, list) and isinstance(right, list) and len(left) == len(right) \
            and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=1e-12, atol=1e-15))
    return left == right


def validate_stored(cfg: dict) -> dict:
    """Recompute every derived field from the stored inputs and require an exact match.

    A stored record must validate against itself: `effective_config` is applied to the
    inputs only, then compared field by field with what was written to disk.
    """
    inputs = {key: value for key, value in cfg.items() if key not in DERIVED}
    # Records created before these fields existed really used r=1 and moving
    # corner vertices. Keep that history explicit instead of relabelling it.
    inputs.setdefault('r0', 1.)
    inputs.setdefault('lid_corner_policy', 'lid')
    fresh = effective_config(inputs)
    mismatched = [key for key in DERIVED if key in cfg and not _same(cfg[key], fresh[key])]
    if mismatched:
        raise ValueError(f'Stored derived fields disagree with the inputs: {mismatched}')
    return fresh

"""Shared persisted adaptive-attempt contract for FEM workflows."""
from typing import Any

import h5py
import numpy as np

FIELDS = ('t', 'step', 'error', 'accepted', 'error_u', 'error_r', 'scalar_defect', 'reason',
          'factor', 'boundary_partition', 'output_truncated', 'previous_accepted_error',
          'previous_accepted_error_u', 'previous_accepted_error_r', 'controller',
          'elapsed_seconds', 'cpu_seconds')
BOOLS = ('accepted', 'boundary_partition', 'output_truncated')
STRINGS = ('reason', 'controller')


def save_attempts(data: Any, attempts: list[dict[str, float | bool | str]]) -> None:
    for key in FIELDS:
        dtype: Any = h5py.string_dtype('utf-8') if key in STRINGS else bool if key in BOOLS else float
        data.create_dataset('attempts/'+key, data=[item[key] for item in attempts], dtype=dtype)


def validate_attempts(data: Any, cfg: dict, times: np.ndarray) -> None:
    if 'attempts/t' not in data or data['attempts/t'].ndim != 1:
        raise ValueError('Missing FEM adaptive attempt records')
    count = data['attempts/t'].shape[0]
    values: dict[str, Any] = {}
    for key in FIELDS:
        path = 'attempts/'+key
        if path not in data or data[path].shape != (count,):
            raise ValueError(f'Missing or invalid FEM attempt field: {key}')
        if key in STRINGS:
            if h5py.check_string_dtype(data[path].dtype) is None:
                raise ValueError('Invalid FEM attempt text type')
            values[key] = data[path].asstr()[:]
        else:
            expected = 'b' if key in BOOLS else 'f'
            if data[path].dtype.kind != expected:
                raise ValueError('Invalid FEM attempt numeric type')
            values[key] = data[path][:]
    for key in ('t', 'step', 'factor', 'elapsed_seconds', 'cpu_seconds'):
        if not np.isfinite(values[key]).all():
            raise ValueError('Nonfinite FEM attempt time/factor')
    if np.any(values['step'] <= 0.) or np.any(values['factor'] <= 0.) \
            or np.any(values['elapsed_seconds'] < -1e-12) or np.any(values['cpu_seconds'] < -1e-12):
        raise ValueError('Invalid FEM attempt step/cost')
    scalar = cfg.get('scalar_atol')
    accepted_index = 0
    previous: dict[str, float] = {'error': float('nan'), 'error_u': float('nan'), 'error_r': float('nan')}
    tolerance = 8*np.finfo(float).eps*max(1., abs(cfg['T']))
    for index in range(count):
        t, h = float(values['t'][index]), float(values['step'][index])
        if t != times[accepted_index] or h > cfg['T']-t+tolerance:
            raise ValueError('FEM adaptive attempted/accepted node mismatch')
        if values['controller'][index] != cfg['controller']['kind']+'Controller':
            raise ValueError('FEM adaptive controller mismatch')
        for key in ('error', 'error_u', 'error_r'):
            history = float(values['previous_accepted_'+key][index])
            if not ((np.isnan(history) and np.isnan(previous[key])) or history == previous[key]):
                raise ValueError('FEM adaptive accepted-error history mismatch')
        accepted = bool(values['accepted'][index])
        error = float(values['error'][index])
        error_u = float(values['error_u'][index])
        error_r = float(values['error_r'][index])
        if accepted:
            forced_floor = cfg.get('min_step_policy') == 'force' and h == cfg['controller']['min_step']
            limit = float('inf') if forced_floor else 1.
            if values['reason'][index] or not np.isfinite([error, error_u]).all() \
                    or not 0. <= error <= limit or not 0. <= error_u <= limit:
                raise ValueError('Invalid accepted FEM adaptive error')
            expected_error = error_u
            if scalar is not None:
                defect = float(values['scalar_defect'][index])
                if not np.isfinite([error_r, defect]).all() or not 0. <= error_r <= limit \
                        or not np.isclose(error_r, abs(defect)/scalar, rtol=1e-12, atol=1e-14):
                    raise ValueError('Invalid FEM adaptive scalar defect')
                expected_error = max(error_u, error_r)
            elif not np.isnan(error_r) or not np.isnan(values['scalar_defect'][index]):
                raise ValueError('Unexpected FEM scalar-control data')
            if error != expected_error or accepted_index+1 >= len(times):
                raise ValueError('FEM adaptive accepted count/error mismatch')
            target = cfg['T'] if h == cfg['T']-t else t+h
            if abs(times[accepted_index+1]-target) > tolerance:
                raise ValueError('FEM adaptive actual step mismatch')
            accepted_index += 1
            previous = {'error': error, 'error_u': error_u, 'error_r': error_r}
        elif not values['reason'][index]:
            raise ValueError('Rejected FEM adaptive trial needs a reason')
    if accepted_index != len(times)-1:
        raise ValueError('FEM adaptive accepted prefix count mismatch')

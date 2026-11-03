"""One configuration for geometry, physical closure, stepping and output."""
from typing import Any

import numpy as np
from experiments.fem_backends import FEM_BACKEND_KEYS, STOKES_BACKENDS, validate_solver_options

from experiments.controllers import controller_settings, make_controller
from solver.adaptivity import ScalarErrorControl
from solver.integrate import step_sizes
from solver.schedule import schedule_nodes
from solver.schemes.registry import scheme_info

DEFAULTS: dict[str, Any] = dict(chord=1., angle=10., gap_ratio=.02, nu=.01, rho=1.,
    pressure_drop=1., ramp_time=.02, convection=True, outlet_treatment='natural',
    bounds=[-3., 6., -2., 2.], surface_points=120, far_size=.4, layer_cells=4,
    quadrature_order=6, scheme='sdirk2_mrsav', gamma=1., r0=0., mode='fixed',
    T=.1, dt=.002, initial_step=.0001, controller=None, scalar_atol=None,
    snapshots=None, log_every=50, stokes_tolerance=1e-9, stokes_backend='direct',
    stokes_ordering='auto', darcy_reference=True)
OPTIONAL = ('wake_region', 'wake_size', 'cache_size', 'inlet_treatment', 'velocity_limit', 'scalar_limit')+FEM_BACKEND_KEYS
METADATA = dict(hele_shaw_schema_version=1, formulation='mwr_parabolic_inertial_brinkman', closure_coefficients=[6/5, 54/35, 6/5, 12.],
    boundary='zero_foil_walls_pressure_traction_ends', pressure_constraint='natural_traction_no_mean_gauge',
    field_layout='full_component_P2_coefficients', airfoil='closed_TE_NACA0012',
    force_semantics='MWR_model_stress_per_span_not_certified_3D_force')


def effective_config(config: dict[str, Any]) -> dict[str, Any]:
    if set(config)-set(DEFAULTS)-set(OPTIONAL):
        raise ValueError(f'Unknown Hele-Shaw keys: {sorted(set(config)-set(DEFAULTS)-set(OPTIONAL))}')
    cfg = {**DEFAULTS, **config}
    validate_solver_options(cfg)
    if cfg.get('inlet_treatment', 'natural') not in ('natural', 'energy_stable_pressure'):
        raise ValueError('Unknown inlet_treatment')
    for key in ('velocity_limit', 'scalar_limit'):
        if key in cfg and (isinstance(cfg[key], bool) or not np.isfinite(cfg[key]) or cfg[key] <= 0.):
            raise ValueError(f'Positive finite {key} required')
    info = scheme_info(cfg['scheme'])
    for key in ['chord', 'gap_ratio', 'nu', 'rho', 'pressure_drop', 'T', 'far_size', 'stokes_tolerance']:
        if isinstance(cfg[key], bool) or not np.isfinite(cfg[key]) or cfg[key] <= 0.:
            raise ValueError(f'Positive {key} required')
    for key in ['convection', 'darcy_reference']:
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    for key, minimum in [('surface_points', 16), ('layer_cells', 1), ('quadrature_order', 5), ('log_every', 1)]:
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f'Invalid {key}')
    if not np.isfinite([cfg['angle'], cfg['gamma'], cfg['ramp_time']]).all() \
            or abs(cfg['angle']) > 45. or cfg['gamma'] < 0. or cfg['ramp_time'] < 0. or cfg['r0'] != 0.:
        raise ValueError('Invalid angle, gamma, ramp_time or r0')
    bounds = np.asarray(cfg['bounds'], dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or not bounds[0] < 0 < 1 < bounds[1] \
            or not bounds[2] < 0 < bounds[3]:
        raise ValueError('Invalid chord-scaled rectangle')
    cfg['bounds'] = bounds.tolist()
    if ('wake_region' in cfg) != ('wake_size' in cfg):
        raise ValueError('wake_region and wake_size must be provided together')
    if 'wake_region' in cfg:
        region = np.asarray(cfg['wake_region'], dtype=float)
        size = cfg['wake_size']
        if region.shape != (4,) or not np.isfinite(region).all() or not region[0] < region[1] \
                or not region[2] < region[3] or not np.isfinite(size) or size <= 0. \
                or region[0] < bounds[0] or region[1] > bounds[1] \
                or region[2] < bounds[2] or region[3] > bounds[3]:
            raise ValueError('Wake box must lie inside the domain and have positive size')
        cfg['wake_region'] = region.tolist()
    if 'cache_size' in cfg and (type(cfg['cache_size']) is not int or cfg['cache_size'] < 1):
        raise ValueError('cache_size must be a positive integer')
    if cfg['outlet_treatment'] not in ('natural', 'backflow_damped') \
            or cfg['stokes_backend'] not in STOKES_BACKENDS \
            or cfg['stokes_ordering'] not in ('auto', 'mmd', 'metis'):
        raise ValueError('Unknown outlet or linear backend')
    if cfg['mode'] == 'fixed':
        if cfg['controller'] is not None or cfg['scalar_atol'] is not None:
            raise ValueError('Controller/scalar_atol require adaptive mode')
        sequence = step_sizes(cfg['T'], dt=cfg['dt'])
        # Kahan summation alone can end at .8999999999999999 for T=.9, dt=.3.
        # Adjust only the last step's roundoff, so stepping and validation agree.
        if schedule_nodes(0., sequence)[-1] != cfg['T']:
            total, correction = 0., 0.
            for step in sequence[:-1]:
                increment = float(step)-correction
                updated = total+increment
                correction = (updated-total)-increment
                total = updated
            sequence[-1] = (cfg['T']-total)+correction
            for _ in range(8):
                end = float(schedule_nodes(0., sequence)[-1])
                if end == cfg['T']:
                    break
                sequence[-1] = np.nextafter(sequence[-1], np.inf if end < cfg['T'] else 0.)
        if schedule_nodes(0., sequence)[-1] != cfg['T'] or sequence[-1] <= 0.:
            raise ValueError('Cannot represent a positive endpoint-aligned final step')
        cfg['actual_steps'] = sequence.tolist()
    elif cfg['mode'] == 'adaptive':
        if 'dt' in config and config['dt'] is not None:
            raise ValueError('Adaptive mode uses initial_step, not dt')
        options = {'kind': 'PI', 'atol': 1e-9, 'rtol': 1e-4,
                   'max_step': .01, **(cfg['controller'] or {})}
        control = make_controller(options, cfg['scheme'])
        cfg['controller'] = controller_settings(control)
        if not np.isfinite(cfg['initial_step']) or cfg['initial_step'] < control.min_step:
            raise ValueError('Invalid initial step')
        if cfg['scalar_atol'] is not None:
            if not cfg['scheme'].endswith('_mrsav'):
                raise ValueError('Scalar control requires SAV')
            ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
        cfg['actual_steps'], cfg['dt'] = None, None
    else:
        raise ValueError('mode must be fixed or adaptive')
    cfg['snapshots'] = [0., cfg['T']] if cfg['snapshots'] is None else list(cfg['snapshots'])
    if not cfg['snapshots'] or any(not np.isfinite(t) or t < 0. or t > cfg['T'] for t in cfg['snapshots']):
        raise ValueError('Invalid snapshot times')
    gap = cfg['gap_ratio']*cfg['chord']
    cfg.update(METADATA, gap=gap, beta=9/7 if cfg['convection'] else 0., drag=10*cfg['nu']/gap**2,
               layer_scale=gap/np.sqrt(10), pressure_scale=1.2*cfg['rho'],
               pressure_semantics=info.pressure_semantics.replace('_mean_zero', '')+'_normalized_MWR_pressure_traction')
    if cfg.get('inlet_treatment') == 'energy_stable_pressure':
        cfg['boundary'] = 'zero_foil_walls_energy_stable_pressure_inlet_'+cfg['outlet_treatment']+'_outlet'
        cfg['pressure_semantics'] += '_energy_stable_inlet'
    return cfg

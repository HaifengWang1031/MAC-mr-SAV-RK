"""Explicit DFG2D-2 physics and fixed/adaptive controller configuration."""
from dataclasses import asdict

import numpy as np

from experiments.periodic_ns.run import make_controller
from solver.adaptivity import ScalarErrorControl
from solver.fem.cylinder import RADIUS, mesh_parameters
from solver.integrate import step_sizes
from solver.schemes.registry import scheme_info
from .representation import METADATA

DEFAULTS = {'resolution': 1., 'circle_segments': None, 'quadrature_order': 6,
            'nu': .001, 'inlet_max': 1.5, 'scheme': 'sdirk3_mrsav', 'gamma': 1., 'r0': 0.,
            'T': 12., 'mode': 'adaptive', 'dt': .002, 'initial_step': .001,
            'controller': None, 'scalar_atol': None, 'snapshots': None,
            'stokes_tolerance': 1e-9, 'stokes_backend': 'direct', 'cache_size': 2, 'log_every': 100}


def effective_config(config: dict) -> dict:
    try:
        return _effective_config(config)
    except (TypeError, KeyError, OverflowError) as error:
        raise ValueError(f'Invalid cylinder configuration: {error}') from error


def _effective_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config)-set(DEFAULTS)-{'steps'}:
        raise ValueError('Unknown cylinder configuration key')
    cfg = {**DEFAULTS, **config}
    cfg['resolution'], cfg['circle_segments'] = mesh_parameters(cfg['resolution'], cfg['circle_segments'])
    if type(cfg['quadrature_order']) is not int or cfg['quadrature_order'] < 5:
        raise ValueError('Cylinder quadrature_order must be an integer >= 5')
    if not isinstance(cfg['scheme'], str):
        raise ValueError('Invalid cylinder scheme')
    info = scheme_info(cfg['scheme'])
    for key in ('nu', 'inlet_max', 'T', 'gamma', 'r0', 'stokes_tolerance'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not np.isfinite(cfg[key]):
            raise ValueError(f'Cylinder {key} must be finite')
    if min(cfg['nu'], cfg['inlet_max'], cfg['T'], cfg['stokes_tolerance']) <= 0 or cfg['gamma'] < 0:
        raise ValueError('Invalid cylinder physical/solver parameters')
    for key in ('cache_size', 'log_every'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'Cylinder {key} must be a positive integer')
    if cfg['stokes_backend'] not in ('direct', 'recycled'):
        raise ValueError('Unknown cylinder Stokes backend')
    if cfg['mode'] == 'fixed':
        if cfg['controller'] is not None or cfg['scalar_atol'] is not None \
                or ('initial_step' in config and config['initial_step'] is not None):
            raise ValueError('Adaptive options require adaptive cylinder mode')
        if 'steps' in config:
            if 'dt' in config and config['dt'] is not None:
                raise ValueError('Use cylinder dt or steps, not both')
            cfg['dt'] = None
        cfg['actual_steps'] = step_sizes(cfg['T'], dt=cfg['dt'], steps=cfg.get('steps')).tolist()
        cfg['initial_step'] = None
    elif cfg['mode'] == 'adaptive':
        if 'steps' in config or ('dt' in config and config['dt'] is not None):
            raise ValueError('Adaptive cylinder uses initial_step, not dt/steps')
        options = {'kind': 'PI', 'atol': 1e-8, 'rtol': 2e-4, 'max_step': .01,
                   'min_factor': .5, 'max_factor': 1.5, 'min_step': 1e-7,
                   **(cfg['controller'] or {})}
        controller = make_controller(options, cfg['scheme'])
        if not np.isfinite(cfg['initial_step']) or cfg['initial_step'] < controller.min_step:
            raise ValueError('Invalid cylinder adaptive initial_step')
        values = asdict(controller)
        values['max_step'] = None if np.isinf(controller.max_step) else controller.max_step
        cfg['controller'] = {'kind': options['kind'], **values}
        cfg['dt'], cfg['actual_steps'] = None, None
        if cfg['scalar_atol'] is not None:
            if not cfg['scheme'].endswith('_mrsav'):
                raise ValueError('Scalar control needs an SAV scheme')
            ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
    else:
        raise ValueError('Cylinder mode must be fixed or adaptive')
    if cfg['snapshots'] is None:
        cfg['snapshots'] = ([0., cfg['T']] if cfg['T'] < 1. else
                            sorted(set([0., *np.linspace(max(0., cfg['T']-2.), cfg['T'], 61).tolist()])))
    else:
        cfg['snapshots'] = list(cfg['snapshots'])
    if any(not np.isfinite(t) or t < 0. or t > cfg['T'] for t in cfg['snapshots']):
        raise ValueError('Cylinder snapshot outside integration interval')
    cfg.update(METADATA, reynolds=(2*cfg['inlet_max']/3)*(2*RADIUS)/cfg['nu'],
               pressure_semantics=info.pressure_semantics.replace('_mean_zero', '_natural_outflow'),
               initial_condition='compatible_steady_stokes_not_impulsive_zero',
               forcing='zero', convection='skew_volume_plus_consistent_boundary_flux')
    return cfg

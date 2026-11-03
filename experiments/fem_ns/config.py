"""FEM experiment settings, independent of computation and record loading."""
import numpy as np
from experiments.controllers import controller_settings, make_controller
from experiments.fem_backends import FEM_BACKEND_KEYS, STOKES_BACKENDS, validate_solver_options
from solver.adaptivity import ScalarErrorControl
from solver.fem import Space
from solver.integrate import step_sizes
from solver.schemes.registry import scheme_info
from .representation import METADATA

DEFAULTS = {'nx': 8, 'ny': 4, 'lx': 2., 'ly': 1., 'quadrature_order': 8,
            'nu': .2, 'amplitude': .1, 'T': .02, 'dt': .01, 'scheme': 'sdirk2_mrsav',
            'gamma': 1., 'r0': 0., 'cache_size': 4, 'stokes_tolerance': 1e-9, 'snapshots': None}

OPTIONS = ('mode', 'initial_step', 'controller', 'scalar_atol', 'stokes_backend', 'stokes_ordering',
           'stokes_factor_backend', 'stokes_threads')+FEM_BACKEND_KEYS


def effective_config(config: dict) -> dict:
    if set(config)-set(DEFAULTS)-{'steps'}-set(OPTIONS):
        raise ValueError('Unknown FEM configuration key')
    cfg = {**DEFAULTS, **config}
    Space(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'], cfg['quadrature_order'])
    if cfg['nx'] == cfg['ny'] == 1:
        raise ValueError('Mesh has too few free velocity DOFs for pressure constraints')
    info = scheme_info(cfg['scheme'])
    for key in ('nu', 'amplitude', 'T', 'stokes_tolerance'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid FEM {key}')
    if not np.isfinite(cfg['gamma']) or cfg['gamma'] < 0 or cfg['r0'] != 0.:
        raise ValueError('FEM MMS requires gamma >= 0 and r0=0')
    if type(cfg['cache_size']) is not int or cfg['cache_size'] < 1:
        raise ValueError('Invalid FEM cache_size')
    mode, backend = cfg.pop('mode', 'fixed'), cfg.get('stokes_backend', 'direct')
    if backend not in STOKES_BACKENDS:
        raise ValueError('Unknown FEM Stokes backend')
    validate_solver_options(cfg)
    if cfg.get('stokes_ordering', 'auto') not in ('auto', 'mmd', 'metis'):
        raise ValueError('Unknown FEM Stokes ordering')
    factor, threads = cfg.get('stokes_factor_backend', 'superlu'), cfg.get('stokes_threads')
    if factor not in ('superlu', 'pardiso') or (factor == 'pardiso' and cfg.get('stokes_ordering', 'auto') != 'auto') \
            or (threads is not None and (type(threads) is not int or threads < 1 or factor != 'pardiso')):
        raise ValueError('Invalid FEM factor backend/thread controls')
    if mode == 'fixed':
        if any(cfg.get(key) is not None for key in ('initial_step', 'controller', 'scalar_atol')):
            raise ValueError('Adaptive FEM options require mode=adaptive')
        for key in OPTIONS:
            cfg.pop(key, None)
        if backend != 'direct':
            cfg['stokes_backend'] = backend
        for key in ('stokes_ordering', 'stokes_factor_backend', 'stokes_threads')+FEM_BACKEND_KEYS:
            if key in config:
                cfg[key] = config[key]
        if 'steps' in config:
            if 'dt' in config:
                raise ValueError('Use dt or steps, not both')
            cfg['dt'] = None
        cfg['actual_steps'] = step_sizes(cfg['T'], dt=cfg['dt'], steps=cfg.get('steps')).tolist()
    elif mode == 'adaptive':
        if 'steps' in config or ('dt' in config and config['dt'] is not None):
            raise ValueError('Adaptive FEM uses initial_step, not dt/steps')
        options = {'kind': 'PI', 'atol': 1e-9, 'rtol': 1e-4, 'max_step': .02,
                   'min_factor': .5, 'max_factor': 1.5, **(cfg.get('controller') or {})}
        controller = make_controller(options, cfg['scheme'])
        initial_step = cfg.get('initial_step', .001)
        if isinstance(initial_step, bool) or not isinstance(initial_step, (int, float)) \
                or not np.isfinite(initial_step) or initial_step < controller.min_step:
            raise ValueError('Invalid FEM adaptive initial_step')
        cfg.update(mode='adaptive', controller=controller_settings(controller), initial_step=initial_step,
                   scalar_atol=cfg.get('scalar_atol'), stokes_backend=backend, actual_steps=None, dt=None)
        if cfg['scalar_atol'] is not None:
            if not cfg['scheme'].endswith('_mrsav'):
                raise ValueError('FEM scalar control requires an SAV scheme')
            ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
    else:
        raise ValueError('FEM mode must be fixed or adaptive')
    cfg['snapshots'] = [0., cfg['T']] if cfg['snapshots'] is None else list(cfg['snapshots'])
    if any(not np.isfinite(t) or t < 0 or t > cfg['T'] for t in cfg['snapshots']):
        raise ValueError('Invalid FEM snapshot times')
    cfg.update(METADATA, case='continuous_polynomial_mms', pressure_semantics=info.pressure_semantics,
               initial_condition='consistent_L2_then_weak_Leray_projection',
               forcing='analytic_ut_plus_advection_minus_nu_laplacian_plus_pressure_gradient')
    return cfg

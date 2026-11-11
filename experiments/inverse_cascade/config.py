"""Central controls in the paper's forcing units; resolve once before execution."""
from typing import Any

import numpy as np

from solver.fourier import FourierSpace

CASE = 'seshasayanan_2014'
SCHEMES = ('sdirk3', 'sdirk3_mrsav')
DEFAULTS: dict[str, Any] = {
    'nx': 128, 'ny': 128, 'kf': 4, 'f0': 1., 'mu': .2,
    'Re_plus': 80., 'Re_minus': 1000., 'gamma': 1.,
    'T': 100., 'dt': .01, 'sample_every': 10, 'log_every': 1000,
    'scheme': 'sdirk3_mrsav', 'seed': 2014, 'noise_rms': .01,
}
# Optional keys are only stored when explicitly supplied, so existing identities keep
# their exact config contents. `coupling_tolerance` gates the frozen SAV coupling
# deviation; it validates discrete self-consistency, not physical accuracy.
OPTIONAL = ('coupling_tolerance',)


def effective_config(config: dict[str, Any]) -> dict[str, Any]:
    """T/dt are in t0=1/sqrt(f0*kf), gamma in 1/t0; L=1, box=(2pi)^2."""
    if unknown := set(config)-set(DEFAULTS)-set(OPTIONAL):
        raise ValueError(f'Unknown cascade keys: {sorted(unknown)}')
    c = {**DEFAULTS, **config}
    tolerance = c.get('coupling_tolerance')
    if tolerance is not None:
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('coupling_tolerance must be positive and finite or null')
        if not c['scheme'].endswith('mrsav'):
            raise ValueError('coupling_tolerance requires an SAV scheme')
        c['coupling_tolerance'] = float(tolerance)
    for key in ('kf', 'sample_every', 'log_every', 'seed'):
        if type(c[key]) is not int or c[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'{key} must be an integer in range')
    space = FourierSpace(c['nx'], c['ny'])
    if min(space.nx, space.ny)/3 <= c['kf']+1:
        raise ValueError('Grid must retain forcing and perturbation modes')
    if c['scheme'] not in SCHEMES:
        raise ValueError('Cascade entry point supports SDIRK3 and SDIRK3-mr-ccSAV')
    positive = ('f0', 'Re_plus', 'Re_minus', 'T', 'dt', 'noise_rms')
    nonnegative = ('mu', 'gamma')
    if not np.isfinite([c[k] for k in (*positive, *nonnegative)]).all() \
            or any(c[k] <= 0 for k in positive) or any(c[k] < 0 for k in nonnegative):
        raise ValueError('Invalid forcing, diffusion, duration, step or perturbation')
    count = c['T']/c['dt']
    if not np.isclose(count, round(count), rtol=0, atol=1e-9) \
            or round(count) < c['sample_every'] or round(count) % c['sample_every']:
        raise ValueError('T must contain whole steps and whole spectral sampling intervals')
    t0 = 1/np.sqrt(c['f0']*c['kf'])
    c.update(case=CASE, t0=float(t0), nu_plus=float(np.sqrt(c['f0'])*c['kf']**(-3.5)/c['Re_plus']),
             nu_minus=float(np.sqrt(c['f0'])*c['kf']**4.5/c['Re_minus']),
             pressure_semantics='leray_eliminated', r0=0., n=2, m=2,
             flux_cutoff='radial_le_K', shell_definition='(K-1,K], zero in shell 0',
             **space.metadata())
    return c

"""Smooth continuous Navier–Stokes MMS on a no-slip rectangle."""
from functools import lru_cache
from typing import Any

import numpy as np
import sympy as sy

from solver.core import State
from solver.fem import FEMModel, Space


@lru_cache(maxsize=16)
def _manufactured(lx: float, ly: float, nu: float, amplitude: float) -> tuple[Any, Any, Any, Any, Any]:
    """Common continuous reference and its exact two temporal force modes."""
    x, y, t = sy.symbols('x y t')
    psi = amplitude*sy.exp(-t)*x**2*(lx-x)**2*y**2*(ly-y)**2
    velocity = (sy.diff(psi, y), -sy.diff(psi, x))
    pressure = amplitude*sy.exp(-t)*(x+2*y-lx/2-ly)
    force = tuple(sy.diff(u, t)+sum(v*sy.diff(u, z) for v, z in zip(velocity, (x, y)))
                  -nu*(sy.diff(u, x, 2)+sy.diff(u, y, 2))+sy.diff(pressure, coordinate)
                  for u, coordinate in zip(velocity, (x, y)))
    quadratic = tuple(sum(v*sy.diff(u, z) for v, z in zip(velocity, (x, y))) for u in velocity)
    linear = tuple((f-q).subs(t, 0) for f, q in zip(force, quadratic))
    nonlinear = tuple(q.subs(t, 0) for q in quadratic)
    return (sy.lambdify((x, y, t), velocity, 'numpy'), sy.lambdify((x, y, t), pressure, 'numpy'),
            sy.lambdify((x, y, t), force, 'numpy'), sy.lambdify((x, y), linear, 'numpy'),
            sy.lambdify((x, y), nonlinear, 'numpy'))


def manufactured(lx: float, ly: float, nu: float, amplitude: float) -> tuple[Any, Any, Any]:
    """Velocity, integral-zero physical pressure and force, not stage-pressure accuracy evidence."""
    return _manufactured(lx, ly, nu, amplitude)[:3]


def build_model(cfg: dict) -> tuple[FEMModel, State]:
    space = Space(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'], cfg['quadrature_order'])
    model = FEMModel(space, cfg['nu'], tolerance=cfg['stokes_tolerance'], cache_size=cfg['cache_size'],
                     stokes_backend=cfg.get('stokes_backend', 'direct'))
    velocity, _, _, linear, nonlinear = _manufactured(cfg['lx'], cfg['ly'], cfg['nu'], cfg['amplitude'])
    initial = model.solve(model.project(lambda x, y: velocity(x, y, 0.)), mass=1., viscosity=0.).velocity
    linear_coefficients, nonlinear_coefficients = model.project(linear), model.project(nonlinear)
    model.force = lambda t: np.exp(-t)*linear_coefficients+np.exp(-2*t)*nonlinear_coefficients
    return model, model.state(0., initial, cfg['r0'])

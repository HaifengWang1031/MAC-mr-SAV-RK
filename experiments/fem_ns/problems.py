"""Smooth continuous Navier–Stokes MMS on a no-slip rectangle."""
from functools import lru_cache
from typing import Any

import numpy as np
import sympy as sy

from solver.core import State
from solver.fem import FEMModel, Space


@lru_cache(maxsize=16)
def manufactured(lx: float, ly: float, nu: float, amplitude: float) -> tuple[Any, Any, Any]:
    """Velocity, integral-zero physical pressure, and continuous body force.

    This reference is not an endpoint-pressure accuracy claim for stage multipliers.
    """
    x, y, t = sy.symbols('x y t')
    psi = amplitude*sy.exp(-t)*x**2*(lx-x)**2*y**2*(ly-y)**2
    velocity = (sy.diff(psi, y), -sy.diff(psi, x))
    pressure = amplitude*sy.exp(-t)*(x+2*y-lx/2-ly)
    force = tuple(sy.diff(u, t)+sum(v*sy.diff(u, z) for v, z in zip(velocity, (x, y)))
                  -nu*(sy.diff(u, x, 2)+sy.diff(u, y, 2))+sy.diff(pressure, coordinate)
                  for u, coordinate in zip(velocity, (x, y)))
    return tuple(sy.lambdify((x, y, t), value, 'numpy') for value in (velocity, pressure, force))


def build_model(cfg: dict) -> tuple[FEMModel, State]:
    space = Space(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'], cfg['quadrature_order'])
    model = FEMModel(space, cfg['nu'], tolerance=cfg['stokes_tolerance'], cache_size=cfg['cache_size'])
    velocity, _, force = manufactured(cfg['lx'], cfg['ly'], cfg['nu'], cfg['amplitude'])
    initial = model.solve(model.project(lambda x, y: velocity(x, y, 0.)), mass=1., viscosity=0.).velocity
    model.force = lambda t: model.project(lambda x, y: force(x, y, t))
    return model, model.state(0., initial, cfg['r0'])

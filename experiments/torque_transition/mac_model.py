"""MAC realization of the same no-slip phase-forced problem; no new time formulas."""
from typing import Any

import numpy as np

from experiments.forced_rotation.model import torque
from solver.mac.grid import MACGrid
from solver.mac.stokes import DirectStokes
from solver.mac.tensor_stokes import TensorStokes
from solver.mac_ns import MACNavierStokes


def make_mac_model(cells: int, phase_degrees: float, backend: str = 'tensor') -> MACNavierStokes:
    """Sample f=(.1 sin(5y), .1 sin(5x+phi)) on free MAC faces."""
    grid = MACGrid(cells, cells, 2*np.pi, 2*np.pi)
    model = MACNavierStokes(grid, 2e-4)
    if backend == 'tensor':
        model.backend = TensorStokes(model.ops, tolerance=1e-10, krylov_tolerance=1e-12)
    elif backend == 'direct':
        model.backend = DirectStokes(model.ops, tolerance=1e-10)
    else:
        raise ValueError('Unknown MAC backend')
    xu, yu = grid.coordinates('u')
    xv, yv = grid.coordinates('v')
    u = np.zeros_like(xu)
    v = np.zeros_like(yv)
    u[:, 1:-1] = .1*np.sin(5*yu[:, 1:-1])
    v[1:-1, :] = .1*np.sin(5*xv[1:-1, :]+np.deg2rad(phase_degrees))
    force = grid.pack(u, v)
    model.force = lambda t: force
    return model


def mac_observables(model: MACNavierStokes, state: Any, case: int) -> dict[str, float]:
    """Area averages, full cell divergence and independently differenced vertex curl.

    Tangential wall velocities use the existing odd ghost reflection. Vertex curl
    integrals use half weights on edges, quarter weights at corners.
    """
    grid = model.grid
    area = grid.lx*grid.ly
    velocity = model.vector(state)
    diagnostic = model.diagnostics(state)
    u, v = grid.unpack(velocity)
    dvdx = np.empty((grid.ny+1, grid.nx+1))
    dudy = np.empty_like(dvdx)
    dvdx[:, 1:-1] = np.diff(v, axis=1)/grid.hx
    dvdx[:, 0], dvdx[:, -1] = 2*v[:, 0]/grid.hx, -2*v[:, -1]/grid.hx
    dudy[1:-1, :] = np.diff(u, axis=0)/grid.hy
    dudy[0, :], dudy[-1, :] = 2*u[0, :]/grid.hy, -2*u[-1, :]/grid.hy
    weights = np.ones_like(dvdx)
    weights[[0, -1], :] *= .5
    weights[:, [0, -1]] *= .5
    enstrophy = .5*grid.area*float(np.sum(weights*(dvdx-dudy)**2))/area
    divergence = (u[:, 1:]-u[:, :-1])/grid.hx+(v[1:, :]-v[:-1, :])/grid.hy
    div2 = grid.area*float(np.sum(divergence**2))/area
    grad2 = diagnostic['h1_seminorm_squared']/area
    return {'energy': diagnostic['kinetic']/area,
            'angular_momentum': diagnostic['angular_momentum']/area,
            'power': diagnostic['power']/area, 'dissipation': diagnostic['dissipation']/area,
            'enstrophy': enstrophy,
            'dissipation_minus_2nuZ': diagnostic['dissipation']/area-2*model.nu*enstrophy,
            'divergence_rms': float(np.sqrt(div2)),
            'relative_divergence': float(np.sqrt(div2/grad2)) if grad2 else 0.,
            'torque': torque(grid, model.force(state.t))}

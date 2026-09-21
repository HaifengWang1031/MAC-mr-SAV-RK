"""Stationary side/bottom walls and a prescribed tangential velocity on the top lid."""
from collections.abc import Callable
import numpy as np
from solver.mac.grid import MACGrid, Array


def lid_viscous_load(grid: MACGrid, nu: float, speed: float,
                     profile: Callable[[Array | float], Array] | None = None) -> Array:
    """Odd extension becomes u_ghost=2*speed-u_top: add 2*nu*speed/hy².

    `profile` scales the lid speed along the wall as a function of the reference coordinate
    xi = 2x/lx - 1, which is how the regularised lid `(1-xi**2)**2` removes the corner
    discontinuity that limits spectral accuracy. The default is the constant sharp lid, so
    existing configurations are unchanged. The profile is evaluated at the interior u faces
    only, which leaves the corner faces at the no-slip value exactly as before.
    """
    if not np.isfinite([nu, speed]).all() or nu <= 0:
        raise ValueError('Finite speed and positive viscosity required')
    u = np.zeros((grid.ny, grid.nx + 1))
    v = np.zeros((grid.ny + 1, grid.nx))
    for j in range(1, grid.nx):
        value = speed if profile is None else speed * profile(2 * (j * grid.hx) / grid.lx - 1)
        u[-1, j] = 2 * nu * value / grid.hy ** 2
    return grid.pack(u, v)


def regularised_lid(ξ: Array | float) -> Array:
    """The corner-free lid profile the spectral tutorial also prefers, in reference units."""
    return (1 - np.asarray(ξ) ** 2) ** 2

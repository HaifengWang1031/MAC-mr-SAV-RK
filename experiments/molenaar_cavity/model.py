"""Physical forcing, MAC sampling and observables; no time integration."""
from collections.abc import Callable
import numpy as np
from solver.mac.grid import MACGrid, Array
from solver.mac.operators import MACOperators
from solver.mac.tensor_stokes import TensorStokes
from solver.mac_ns import MACNavierStokes


def force_field(grid: MACGrid, *, A0: float, kappa: float, q_spin: float) -> Array:
    if grid.lx != 2 or grid.ly != 2 or kappa <= 0 or A0 <= 0:
        raise ValueError('Require the side-2 square and positive amplitude/wavenumber')
    if not np.isfinite([A0, kappa, q_spin]).all():
        raise ValueError('Nonfinite force parameter')
    _,yu = grid.coordinates('u')
    xv,_ = grid.coordinates('v')
    fu = -2*A0/kappa*np.sin(kappa*(yu-1)) - q_spin*(yu-1)/2
    fv = 2*A0/kappa*np.sin(kappa*(xv-1)) + q_spin*(xv-1)/2
    fu[:,[0,-1]] = 0
    fv[[0,-1],:] = 0
    return grid.pack(fu,fv)


def make_model(config: dict) -> MACNavierStokes:
    grid = MACGrid(config['n'],config['n'],2.,2.)
    packed = force_field(grid,A0=config['A0'],kappa=config['kappa'],q_spin=config['q_spin'])
    force: Callable[[float],Array] = lambda t: packed.copy()
    backend = (TensorStokes(MACOperators(grid),tolerance=1e-9,krylov_tolerance=1e-12)
               if config['backend']=='tensor' else None)
    return MACNavierStokes(grid,config['A0']/config['nu_prime'],force=force,backend=backend)


def probe_vorticity(model: MACNavierStokes, velocity: Array) -> float:
    """Bilinear interior-vertex curl at physical (X,Y)=(1/2,0)."""
    g = model.grid
    u,v = g.unpack(velocity)
    omega = (v[1:-1,1:]-v[1:-1,:-1])/g.hx-(u[1:,1:-1]-u[:-1,1:-1])/g.hy
    xs = np.arange(1,g.nx)*g.hx
    ys = np.arange(1,g.ny)*g.hy
    return float(np.interp(1.,ys,[np.interp(1.5,xs,row) for row in omega]))

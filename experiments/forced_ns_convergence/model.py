"""Prescribed physical forcing and initial data; no manufactured exact solution."""
import numpy as np
from solver.mac.grid import MACGrid, Array


def force_vector(grid: MACGrid, amplitude: float = 1.) -> Array:
    x, _ = grid.coordinates('v')
    # Body force is sampled only at free faces; it has no velocity boundary constraint.
    return np.concatenate([np.zeros(grid.nu), (amplitude*np.sin(x[1:-1,:])).ravel()])


def force_vector_cos(grid: MACGrid, amplitude: float = 1.) -> Array:
    """Horizontal body force `f_x = amplitude*cos(x)`, sampled at the free u faces.

    The borrowed benchmark states its external force as the scalar `cos(x)` added to the
    vorticity equation; in a velocity-pressure formulation the closest reading is a body force
    in the x direction, and it is sampled only where the discrete momentum equation is imposed
    (interior u faces), so no wall constraint is touched (`docs/validation.md` records the
    reading and its difference from the vorticity form).
    """
    x, _ = grid.coordinates('u')
    force_u = np.zeros((grid.ny, grid.nx+1))
    force_v = np.zeros((grid.ny+1, grid.nx))
    force_u[:, 1:-1] = amplitude*np.cos(x[:, 1:-1])
    return grid.pack(force_u, force_v)


def _trig_stream(x, y, lx, ly, modes: int = 10):
    """Stream function and its first derivatives: the benchmark shape times a wall weight.

    The benchmark's initial vorticity
    `omega_0 = sum_{k,m=1}^{10} (k^2+m^2)^(-3/2) cos(kx) cos(my)` is used as the shape of a
    stream function. Since `u = d_y psi` and `v = -d_x psi`, a psi that merely vanishes on the
    wall still leaves a tangential velocity there, so the shape is multiplied by
    `W = (x/lx)^2(1-x/lx)^2 (y/ly)^2(1-y/ly)^2`: `psi = 0` together with `d_n psi = 0` on all
    four walls, which is what homogeneous no-slip requires. This weight is our adaptation of
    the benchmark, which is periodic and has no wall at all; it also raises the initial
    spectral content from 10 to about 16 modes per direction.
    """
    k = np.arange(1, modes+1)
    m = np.arange(1, modes+1)
    c = 1.0/(k[:, None]**2 + m[None, :]**2)**1.5
    ax = 2*np.pi*k/lx
    by = 2*np.pi*m/ly
    cos_x = np.cos(ax[:, None]*x)
    sin_x = np.sin(ax[:, None]*x)
    cos_y = np.cos(by[:, None]*y)
    sin_y = np.sin(by[:, None]*y)
    shape = np.einsum('km,kn,ml->ln', c, cos_x, cos_y)
    dshape_dx = np.einsum('km,kn,ml->ln', c, -ax[:, None]*sin_x, cos_y)
    dshape_dy = np.einsum('km,kn,ml->ln', c, cos_x, -by[:, None]*sin_y)
    wx = (x/lx)**2*(1-x/lx)**2
    wy = (y/ly)**2*(1-y/ly)**2
    dwx = 2*(x/lx)*(1-x/lx)*(1-2*x/lx)/lx
    dwy = 2*(y/ly)*(1-y/ly)*(1-2*y/ly)/ly
    # Everything is indexed [y, x]: the weight factors are a column in y and a row in x.
    weight = wy[:, None]*wx[None, :]
    psi = weight*shape
    dpsi_dx = wy[:, None]*dwx[None, :]*shape + weight*dshape_dx
    dpsi_dy = dwy[:, None]*wx[None, :]*shape + weight*dshape_dy
    return psi, dpsi_dx, dpsi_dy


def initial_velocity_trig(grid: MACGrid, modes: int = 10) -> Array:
    """Homogeneous no-slip velocity from the benchmark's trigonometric stream shape.

    Divergence-free by construction up to the sampling error, which is `O(h^2)`: the first
    Stokes stage imposes the discrete constraint, so the state is discretely solenoidal from
    then on. The amplitude of the benchmark's coefficients is kept as given, with no rescaling.
    """
    x_u, y_u = grid.coordinates('u')
    x_v, y_v = grid.coordinates('v')
    _, _, dpsi_dy = _trig_stream(x_u[0, :], y_u[:, 0], grid.lx, grid.ly, modes)
    _, dpsi_dx, _ = _trig_stream(x_v[0, :], y_v[:, 0], grid.lx, grid.ly, modes)
    velocity_u = dpsi_dy
    velocity_v = -dpsi_dx
    # The no-slip weight W <= 1/256 makes the raw field far too small to study temporal error
    # against roundoff, so the field is normalised to max(|u|,|v|) = 1. The benchmark's
    # coefficients fix the shape, not the amplitude of a stream function it never formed;
    # the normalisation is recorded in docs/validation.md.
    scale = 1.0/max(np.abs(velocity_u).max(), np.abs(velocity_v).max())
    velocity_u = scale*velocity_u
    velocity_v = scale*velocity_v
    velocity_u[:, [0, -1]] = 0.
    velocity_u[[0, -1], :] = 0.
    velocity_v[[0, -1], :] = 0.
    velocity_v[:, [0, -1]] = 0.
    return grid.pack(velocity_u, velocity_v)

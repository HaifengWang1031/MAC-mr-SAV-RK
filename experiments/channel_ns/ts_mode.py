"""Tollmien--Schlichting seeds and an independent continuous stability reference.

The seed is an eigenvector of the linearised *same MAC equations* in the first
periodic Fourier mode. A vertex streamfunction spans its divergence-free space.
This is an experiment initial condition, not a replacement time/space solver.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import eig

from solver.channel import ChannelGrid
from solver.channel.kernels import convection
from solver.mac.grid import Array


def continuous_eigenvalue(reynolds: float, alpha: float, degree: int = 80) -> complex:
    """Clamped Chebyshev Orr--Sommerfeld reference on [-1,1], U=1-y².

Returns the temporal eigenvalue lambda: exp(lambda*t+i*alpha*x). Four boundary
constraints are eliminated before enforcing interior collocation equations.
At Re=10000, alpha=1 the known value is .003739670623-.237526488820j.
"""
    if min(reynolds, alpha) <= 0 or degree < 24:
        raise ValueError('Require positive Re/alpha and degree >= 24')
    y = np.cos(np.pi*np.arange(degree+1)/degree)
    c = np.r_[2., np.ones(degree-1), 2.]*(-1.)**np.arange(degree+1)
    differences = y[:, None]-y[None, :]
    derivative = (c[:, None]/c[None, :])/(differences+np.eye(degree+1))
    derivative -= np.diag(derivative.sum(axis=1))
    rows = np.arange(2, degree-1)
    extension = np.zeros((degree+1, degree-3))
    extension[rows, np.arange(degree-3)] = 1.
    extension[[1, degree-1]] = np.linalg.solve(
        derivative[[0, degree]][:, [1, degree-1]], -derivative[[0, degree]][:, rows])
    laplacian = derivative@derivative-alpha**2*np.eye(degree+1)
    operator = laplacian@laplacian/reynolds \
        -1j*alpha*((1-y*y)[:, None]*laplacian)-2j*alpha*np.eye(degree+1)
    values = eig((operator@extension)[rows], (laplacian@extension)[rows], right=False)
    physical = values[np.isfinite(values) & (values.imag < 0.) & (values.imag > -alpha)
                      & (values.real < .1)]
    if not len(physical):
        raise RuntimeError('No physical Orr--Sommerfeld eigenvalue found')
    return complex(physical[np.argmax(physical.real)])


@dataclass(frozen=True)
class TSMode:
    eigenvalue: complex
    streamfunction: Array
    velocity: Array
    eigen_residual: float
    reynolds: float
    alpha: float
    centreline_speed: float

    def metadata(self) -> dict[str, float | str]:
        return {'eigenvalue_real': self.eigenvalue.real, 'eigenvalue_imag': self.eigenvalue.imag,
                'eigen_residual': self.eigen_residual, 'reynolds_half_height': self.reynolds,
                'alpha_half_height': self.alpha, 'centreline_speed': self.centreline_speed,
                'normalisation': 'initial real velocity maximum magnitude equals one',
                'seed': 'linearised MAC divergence-free fundamental eigenmode'}


@lru_cache(maxsize=12)
def discrete_mode(grid: ChannelGrid, nu: float, driving: float) -> TSMode:
    """Leading fundamental MAC eigenmode; factors are not used during integration.

The quadratic convection derivative is exact under the symmetric difference,
up to roundoff. Z spans all solenoidal velocities in this nonzero x mode, so
Z*Z and Z*(nu*K+C) give the constrained linearised dynamics without pressure.
"""
    if nu <= 0 or driving <= 0:
        raise ValueError('TS experiment requires positive viscosity and driving')
    ny, nx = grid.ny, grid.nx
    extension = np.zeros((ny+1, ny-1))
    extension[1:-1] = np.eye(ny-1)
    theta = 2*np.pi/nx
    symbol = np.expm1(1j*theta)/grid.hx
    basis = np.vstack([np.diff(extension, axis=0)/grid.hy, -symbol*extension[1:-1]])
    x_laplacian = 4*np.sin(theta/2)**2/grid.hx**2
    ku = np.diag(np.full(ny, 2/grid.hy**2+x_laplacian)) \
        +np.diag(np.full(ny-1, -1/grid.hy**2), 1)+np.diag(np.full(ny-1, -1/grid.hy**2), -1)
    ku[[0, -1], [0, -1]] = 3/grid.hy**2+x_laplacian
    kv = np.diag(np.full(ny-1, 2/grid.hy**2+x_laplacian)) \
        +np.diag(np.full(ny-2, -1/grid.hy**2), 1)+np.diag(np.full(ny-2, -1/grid.hy**2), -1)
    _, y = grid.coordinates('u')
    base_u = driving*(y*(grid.ly-y)/(2*nu)+grid.hy**2/(8*nu))
    derivative = np.empty_like(basis, dtype=complex)
    phase = np.cos(theta*np.arange(nx+1))
    phase[-1] = phase[0]
    delta = 1e-4*grid.hy
    for column in range(ny-1):
        psi = extension[:, column, None]*phase
        u, v = grid.unpack(grid.curl(psi))
        positive_u, positive_v = convection(base_u+delta*u, delta*v, grid.hx, grid.hy)
        negative_u, negative_v = convection(base_u-delta*u, -delta*v, grid.hx, grid.hy)
        derivative[:, column] = np.r_[
            2*np.fft.rfft((positive_u-negative_u)[:, :-1]/(2*delta), axis=1)[:, 1]/nx,
            2*np.fft.rfft((positive_v-negative_v)[1:-1]/(2*delta), axis=1)[:, 1]/nx]
    mass = basis.conj().T@basis
    operator = -nu*basis.conj().T@np.vstack([ku@basis[:ny], kv@basis[ny:]]) \
        -basis.conj().T@derivative
    values, vectors = eig(operator, mass)
    index = int(np.argmax(values.real))
    value, coefficients = complex(values[index]), vectors[:, index]
    if not np.isfinite(value):
        raise RuntimeError('Nonfinite MAC instability eigenvalue')
    residual = float(np.linalg.norm(operator@coefficients-value*(mass@coefficients))
                     /(1+np.linalg.norm(operator@coefficients)+abs(value)*np.linalg.norm(mass@coefficients)))
    psi_y = extension@coefficients
    # The centre phase gives reproducible initial fields across refinements.
    psi_y *= np.exp(-1j*np.angle(psi_y[ny//2]))
    psi = np.real(psi_y[:, None]*np.exp(1j*theta*np.arange(nx+1)))
    psi[:, -1] = psi[:, 0]
    velocity = grid.curl(psi)
    u, v = grid.unpack(velocity)
    peak = float(np.max(np.hypot(.5*(u[:, :-1]+u[:, 1:]), .5*(v[:-1]+v[1:]))))
    psi, velocity = psi/peak, velocity/peak
    speed = driving*grid.ly**2/(8*nu)
    for array in (psi, velocity):
        array.setflags(write=False)
    return TSMode(value, psi, velocity, residual, speed*grid.ly/(2*nu),
                  np.pi*grid.ly/grid.lx, speed)

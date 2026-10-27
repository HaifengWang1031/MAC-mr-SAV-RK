"""Uniform MAC faces with periodic x and stationary no-slip y walls."""
from dataclasses import dataclass

import numpy as np

from ..mac.grid import Array


@dataclass(frozen=True)
class ChannelGrid:
    nx: int
    ny: int
    lx: float = 1.
    ly: float = 1.

    def __post_init__(self) -> None:
        if type(self.nx) is not int or type(self.ny) is not int \
                or min(self.nx, self.ny) < 3:
            raise ValueError('Require integer nx,ny >= 3')
        if not np.isfinite([self.lx, self.ly]).all() or min(self.lx, self.ly) <= 0:
            raise ValueError('Require positive finite domain lengths')

    @property
    def hx(self) -> float: return self.lx/self.nx
    @property
    def hy(self) -> float: return self.ly/self.ny
    @property
    def nu(self) -> int: return self.nx*self.ny
    @property
    def nv(self) -> int: return self.nx*(self.ny-1)
    @property
    def size(self) -> int: return self.nu+self.nv
    @property
    def np(self) -> int: return self.nx*self.ny
    @property
    def area(self) -> float: return self.hx*self.hy

    def coordinates(self, field: str) -> tuple[Array, Array]:
        if field == 'u':
            x, y = np.arange(self.nx+1)*self.hx, (np.arange(self.ny)+.5)*self.hy
        elif field == 'v':
            x, y = (np.arange(self.nx)+.5)*self.hx, np.arange(self.ny+1)*self.hy
        elif field == 'p':
            x, y = (np.arange(self.nx)+.5)*self.hx, (np.arange(self.ny)+.5)*self.hy
        elif field == 'vertex':
            x, y = np.arange(self.nx+1)*self.hx, np.arange(self.ny+1)*self.hy
        else:
            raise ValueError(f'Unknown field {field}')
        return np.meshgrid(x, y)

    def _faces(self, u: Array, v: Array) -> None:
        if u.shape != (self.ny, self.nx+1) or v.shape != (self.ny+1, self.nx):
            raise ValueError('Incorrect channel MAC array shapes')
        if np.iscomplexobj(u) or np.iscomplexobj(v) or not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError('Channel faces must be real and finite')
        tolerance = 32*np.finfo(float).eps*max(1., float(np.max(np.abs(u))))
        if not np.allclose(u[:, 0], u[:, -1], rtol=0., atol=tolerance):
            raise ValueError('Streamwise u faces must be periodic')

    def pack(self, u: Array, v: Array) -> Array:
        self._faces(u, v)
        if np.any(v[[0, -1], :] != 0):
            raise ValueError('Normal wall velocities must be zero')
        return self.pack_load(u, v)

    def pack_load(self, u: Array, v: Array) -> Array:
        """Sample loads on free faces; physical force at a wall need not vanish."""
        self._faces(u, v)
        return np.concatenate((u[:, :self.nx].ravel(), v[1:-1, :].ravel()))

    def unpack(self, velocity: Array) -> tuple[Array, Array]:
        if velocity.shape != (self.size,) or np.iscomplexobj(velocity) or not np.isfinite(velocity).all():
            raise ValueError('Incorrect or nonfinite channel free vector')
        u = np.empty((self.ny, self.nx+1))
        u[:, :self.nx] = velocity[:self.nu].reshape(self.ny, self.nx)
        u[:, -1] = u[:, 0]
        v = np.zeros((self.ny+1, self.nx))
        v[1:-1, :] = velocity[self.nu:].reshape(self.ny-1, self.nx)
        return u, v

    def inner(self, a: Array, b: Array) -> float:
        return float(self.area*np.dot(a, b))

    def norm(self, a: Array) -> float:
        return float(np.sqrt(self.inner(a, a)))

    def curl(self, psi: Array) -> Array:
        """Discrete vertex curl; each wall streamfunction must be constant in x."""
        if psi.shape != (self.ny+1, self.nx+1) or np.iscomplexobj(psi) or not np.isfinite(psi).all():
            raise ValueError('Incorrect or nonfinite vertex streamfunction')
        tolerance = 32*np.finfo(float).eps*max(1., float(np.max(np.abs(psi))))
        if not np.allclose(psi[:, 0], psi[:, -1], rtol=0., atol=tolerance) \
                or np.max(np.ptp(psi[[0, -1], :], axis=1)) > tolerance:
            raise ValueError('Require periodic psi and constant wall psi')
        canonical = psi.copy()
        canonical[:, -1] = canonical[:, 0]
        canonical[0, :] = canonical[0, 0]
        canonical[-1, :] = canonical[-1, 0]
        return self.pack(np.diff(canonical, axis=0)/self.hy,
                         -np.diff(canonical, axis=1)/self.hx)

    def metadata(self) -> dict[str, str]:
        return {'boundary': 'periodic_x_no_slip_y', 'discretization': 'mac_second_order',
                'field_layout': 'staggered_mac_periodic_x', 'periodic_u_endpoint': 'duplicated'}

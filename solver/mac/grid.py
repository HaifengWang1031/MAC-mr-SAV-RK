"""Uniform rectangular MAC layout; full face arrays, interior free velocities."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

@dataclass(frozen=True)
class MACGrid:
    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.nx, bool) or isinstance(self.ny, bool) or not isinstance(self.nx, int) or not isinstance(self.ny, int):
            raise ValueError('nx, ny must be integers')
        if min(self.nx, self.ny) < 3 or not np.isfinite([self.lx, self.ly]).all() or min(self.lx, self.ly) <= 0:
            raise ValueError('Require nx,ny >= 3 and positive finite domain lengths')

    @property
    def hx(self) -> float: return self.lx / self.nx
    @property
    def hy(self) -> float: return self.ly / self.ny
    @property
    def nu(self) -> int: return (self.nx - 1) * self.ny
    @property
    def nv(self) -> int: return self.nx * (self.ny - 1)
    @property
    def size(self) -> int: return self.nu + self.nv
    @property
    def np(self) -> int: return self.nx * self.ny
    @property
    def area(self) -> float: return self.hx * self.hy

    def coordinates(self, field: str) -> tuple[Array, Array]:
        x: Array
        y: Array
        if field == 'u':
            x, y = np.arange(self.nx + 1,dtype=float)*self.hx, (np.arange(self.ny,dtype=float)+0.5)*self.hy
        elif field == 'v':
            x, y = (np.arange(self.nx,dtype=float)+0.5)*self.hx, np.arange(self.ny+1,dtype=float)*self.hy
        elif field == 'p':
            x, y = (np.arange(self.nx,dtype=float)+0.5)*self.hx, (np.arange(self.ny,dtype=float)+0.5)*self.hy
        elif field == 'vertex':
            x, y = np.arange(self.nx+1,dtype=float)*self.hx, np.arange(self.ny+1,dtype=float)*self.hy
        else:
            raise ValueError(f'Unknown field {field}')
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    def pack(self, u: Array, v: Array) -> Array:
        if u.shape != (self.ny,self.nx+1) or v.shape != (self.ny+1,self.nx):
            raise ValueError('Incorrect MAC array shapes')
        if np.any(u[:,[0,-1]] != 0) or np.any(v[[0,-1],:] != 0):
            raise ValueError('Normal boundary velocities must be zero')
        return np.concatenate((u[:,1:-1].ravel(), v[1:-1,:].ravel()))

    def unpack(self, velocity: Array) -> tuple[Array, Array]:
        if velocity.shape != (self.size,): raise ValueError('Incorrect free-vector shape')
        u = np.zeros((self.ny,self.nx+1))
        v = np.zeros((self.ny+1,self.nx))
        u[:,1:-1] = velocity[:self.nu].reshape(self.ny,self.nx-1)
        v[1:-1,:] = velocity[self.nu:].reshape(self.ny-1,self.nx)
        return u,v

    def inner(self, a: Array, b: Array) -> float:
        return float(self.area * np.dot(a,b))

    def norm(self, a: Array) -> float:
        return float(np.sqrt(self.inner(a,a)))

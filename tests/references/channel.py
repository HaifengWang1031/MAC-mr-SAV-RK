"""Test-owned Channel MMS, without experiment builders or configuration.

Manufactured uses continuous PDE derivatives for spatial tests;
DiscreteManufactured uses explicit NumPy stencils for fixed-grid temporal tests,
never the model's apply_K/nonlinear. These are different validation claims.
"""
import numpy as np

from solver.channel import ChannelGrid
from solver.mac.grid import Array


class Manufactured:
    """psi=a(t)[sin(kx)+.2sin(2kx)]sin²(pi*y/H), with nonzero pressure.

    Continuous derivatives supply forcing independently of the MAC operators.
    Initial data use the vertex curl to satisfy discrete continuity exactly.
    """
    def __init__(self, grid: ChannelGrid, nu: float, amplitude: float = .1) -> None:
        self.grid, self.nu, self.scale = grid, nu, amplitude
        self.base, self.stiffness, self.convection, self.gradient = self._samples()
        x, y = grid.coordinates('vertex')
        psi = (np.sin(2*np.pi*x/grid.lx)+.2*np.sin(4*np.pi*x/grid.lx))*np.sin(np.pi*y/grid.ly)**2
        psi[[0, -1]] = 0.
        self.initial_shape = grid.curl(psi)

    def amplitude(self, t: float) -> float:
        return float(self.scale*np.exp(-t)*(1+.25*np.sin(np.pi*t)))

    def derivative(self, t: float) -> float:
        return float(self.scale*np.exp(-t)*(-1-.25*np.sin(np.pi*t)+.25*np.pi*np.cos(np.pi*t)))

    def _samples(self) -> tuple[Array, Array, Array, Array]:
        grid = self.grid
        k, b = 2*np.pi/grid.lx, np.pi/grid.ly
        samples: list[list[Array]] = [[], [], [], []]
        for component in ('u', 'v'):
            x, y = grid.coordinates(component)
            s = np.sin(k*x)+.2*np.sin(2*k*x)
            sx = k*(np.cos(k*x)+.4*np.cos(2*k*x))
            sxx = -k*k*(np.sin(k*x)+.8*np.sin(2*k*x))
            sxxx = -k**3*(np.cos(k*x)+1.6*np.cos(2*k*x))
            phi = np.sin(b*y)**2
            py = b*np.sin(2*b*y)
            pyy = 2*b*b*np.cos(2*b*y)
            pyyy = -4*b**3*np.sin(2*b*y)
            if component == 'u':
                values = [s*py, -sxx*py-s*pyyy, s*sx*(py*py-phi*pyy),
                          -k*np.sin(k*x)*np.cos(b*y)]
                for value in values: value[:, -1] = value[:, 0]
            else:
                values = [-sx*phi, sxxx*phi+sx*pyy, (sx*sx-s*sxx)*phi*py,
                          -b*np.cos(k*x)*np.sin(b*y)]
                values[0][[0, -1]] = 0.
            for collection, value in zip(samples, values): collection.append(value)
        packed = [grid.pack_load(*components) for components in samples]
        return packed[0], packed[1], packed[2], packed[3]

    def exact(self, t: float) -> Array:
        return self.amplitude(t)*self.base

    def force(self, t: float) -> Array:
        a = self.amplitude(t)
        return self.derivative(t)*self.base+self.nu*a*self.stiffness+a*a*self.convection+a*self.gradient


def reference_flux(u: Array, v: Array, hx: float, hy: float) -> tuple[Array, Array]:
    """Independent vectorised dual-cell flux evaluation for temporal MMS."""
    q = u[:, :-1]
    east, west = .5*(q+np.roll(q, -1, 1)), .5*(q+np.roll(q, 1, 1))
    north, south = .5*(v[1:]+np.roll(v[1:], 1, 1)), .5*(v[:-1]+np.roll(v[:-1], 1, 1))
    padded = np.vstack([-q[:1], q, -q[-1:]])
    cu = (east*.5*(q+np.roll(q, -1, 1))-west*.5*(q+np.roll(q, 1, 1)))/hx
    cu += (north*.5*(q+padded[2:])-south*.5*(q+padded[:-2]))/hy
    cu -= .5*q*((east-west)/hx+(north-south)/hy)
    qv = v[1:-1]
    west = .5*(q[:-1]+q[1:]); east = np.roll(west, -1, 1)
    north, south = .5*(qv+v[2:]), .5*(qv+v[:-2])
    cv = (east*.5*(qv+np.roll(qv, -1, 1))-west*.5*(qv+np.roll(qv, 1, 1)))/hx
    cv += (north*.5*(qv+v[2:])-south*.5*(qv+v[:-2]))/hy
    cv -= .5*qv*((east-west)/hx+(north-south)/hy)
    return np.column_stack([cu, cu[:, 0]]), np.vstack([np.zeros_like(v[:1]), cv, np.zeros_like(v[:1])])


class DiscreteManufactured(Manufactured):
    """Fixed-grid MMS: NumPy stencils supply f, never model.apply_K/nonlinear."""
    def __init__(self, grid: ChannelGrid, nu: float, amplitude: float = .1) -> None:
        super().__init__(grid, nu, amplitude)
        self.base = self.initial_shape
        u, v = grid.unpack(self.base)
        q = u[:, :-1]
        pad = np.vstack([-q[:1], q, -q[-1:]])
        ku = (2*q-np.roll(q, 1, 1)-np.roll(q, -1, 1))/grid.hx**2 \
            +(2*q-pad[:-2]-pad[2:])/grid.hy**2
        qv = v[1:-1]
        kv = (2*qv-np.roll(qv, 1, 1)-np.roll(qv, -1, 1))/grid.hx**2 \
            +(2*qv-v[:-2]-v[2:])/grid.hy**2
        self.stiffness = np.concatenate([ku.ravel(), kv.ravel()])
        self.convection = grid.pack(*reference_flux(u, v, grid.hx, grid.hy))
        x, y = grid.coordinates('p')
        p = np.cos(2*np.pi*x/grid.lx)*np.cos(np.pi*y/grid.ly)
        gx = (p-np.roll(p, 1, 1))/grid.hx
        gy = np.diff(p, axis=0)/grid.hy
        self.gradient = np.concatenate([gx.ravel(), gy.ravel()])

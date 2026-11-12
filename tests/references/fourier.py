"""Test-owned analytic Fourier references, independent of task configuration.

Forcing uses continuous derivatives, not numerical convection or Stokes solves.
"""
import numpy as np

from solver.fourier import FourierSpace
from solver.fourier.space import RealArray


class Manufactured:
    """psi=a(t)[sin(X)+.2sin(2X)]sin(Y), p=a(t)cos(X)cos(Y).

    Physical derivatives below are analytic; no numerical operator supplies f.
    Nonlinear products have |mx|<=4, |my|<=2. nx>=15,ny>=9 resolves the
    continuous forcing inside the strict band (smaller grids approximate it).
    """
    def __init__(self, space: FourierSpace, nu: float, amplitude: float = 1.) -> None:
        self.space, self.nu, self.scale = space, nu, amplitude
        ax, by = 2*np.pi/space.lx, 2*np.pi/space.ly
        x, y = ax*space.x, by*space.y
        self.base = np.stack([by*(np.sin(x)+.2*np.sin(2*x))*np.cos(y),
                              -ax*(np.cos(x)+.4*np.cos(2*x))*np.sin(y)])
        ux = ax*by*(np.cos(x)+.4*np.cos(2*x))*np.cos(y)
        uy = -by*by*(np.sin(x)+.2*np.sin(2*x))*np.sin(y)
        vx = ax*ax*(np.sin(x)+.8*np.sin(2*x))*np.sin(y)
        vy = -ax*by*(np.cos(x)+.4*np.cos(2*x))*np.cos(y)
        u, v = self.base
        self.convection = np.stack([u*ux+v*uy, u*vx+v*vy])
        self.negative_laplacian = np.stack([
            by*((ax*ax+by*by)*np.sin(x)+.2*(4*ax*ax+by*by)*np.sin(2*x))*np.cos(y),
            -ax*((ax*ax+by*by)*np.cos(x)+.4*(4*ax*ax+by*by)*np.cos(2*x))*np.sin(y)])
        self.pressure_gradient = np.stack([-ax*np.sin(x)*np.cos(y),
                                           -by*np.cos(x)*np.sin(y)])

    def amplitude(self, t: float) -> float:
        return float(self.scale*np.exp(-t)*(1+.25*np.sin(t)))

    def derivative(self, t: float) -> float:
        return float(self.scale*np.exp(-t)*(-1-.25*np.sin(t)+.25*np.cos(t)))

    def exact(self, t: float) -> RealArray:
        return self.amplitude(t)*self.base

    def force(self, t: float) -> RealArray:
        a = self.amplitude(t)
        return self.derivative(t)*self.base+self.nu*a*self.negative_laplacian \
            +a*a*self.convection+a*self.pressure_gradient


def smooth_stokes(space: FourierSpace) -> tuple[RealArray, RealArray]:
    """Infinite-modal steady Stokes reference: psi=exp(.7cos X+.4cos Y).

    Returns exact real velocity and -Delta velocity; steady pressure is zero.
    """
    ax, by = 2*np.pi/space.lx, 2*np.pi/space.ly
    x, y = ax*space.x, by*space.y
    psi = np.exp(.7*np.cos(x)+.4*np.cos(y))
    qx, qy = -.7*ax*np.sin(x), -.4*by*np.sin(y)
    qxx, qyy = -.7*ax*ax*np.cos(x), -.4*by*by*np.cos(y)
    qxxx, qyyy = .7*ax**3*np.sin(x), .4*by**3*np.sin(y)
    velocity = np.stack([psi*qy, -psi*qx])
    negative_laplacian = np.stack([
        -psi*(qy*(qx*qx+qxx+qy*qy+3*qyy)+qyyy),
        psi*(qx*(qy*qy+qyy+qx*qx+3*qxx)+qxxx)])
    return velocity, negative_laplacian

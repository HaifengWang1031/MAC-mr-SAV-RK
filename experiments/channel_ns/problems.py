"""Channel references: continuous spatial MMS and independent discrete temporal MMS."""
import numpy as np

from solver.channel import ChannelGrid, ChannelModel
from solver.core import State
from solver.mac.grid import Array
from solver.schemes.registry import SCHEMES, make_scheme as scheme
CASES = ('poiseuille', 'startup', 'perturbed', 'ts_wave', 'mms', 'discrete_mms')


def poiseuille(grid: ChannelGrid, nu: float, driving: float, *, discrete: bool = False) -> Array:
    _, y = grid.coordinates('u')
    u = driving*y*(grid.ly-y)/(2*nu)
    if discrete: u += driving*grid.hy**2/(8*nu)
    return grid.pack(u, np.zeros((grid.ny+1, grid.nx)))


def startup(grid: ChannelGrid, nu: float, driving: float, t: float) -> Array:
    """Continuous start from rest; diffusion sine series at actual physical time.

    t=0 is returned exactly. For t>0 the exponential suppresses high modes;
    truncation is chosen from a 1e-15 exponential threshold, not the MAC grid.
    """
    if t == 0: return np.zeros(grid.size)
    _, y = grid.coordinates('u')
    count = max(1, int(np.ceil(np.sqrt(-np.log(1e-15)*grid.ly**2/(nu*np.pi**2*t)))))
    transient = np.zeros_like(y)
    for n in range(1, count+1, 2):
        transient += (4*driving*grid.ly**2/(nu*np.pi**3*n**3)
                      *np.exp(-nu*(n*np.pi/grid.ly)**2*t)*np.sin(n*np.pi*y/grid.ly))
    transient[:, -1] = transient[:, 0]
    return poiseuille(grid, nu, driving)-grid.pack(transient, np.zeros((grid.ny+1, grid.nx)))


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


def build_model(config: dict) -> tuple[ChannelModel, State, Manufactured | None]:
    grid = ChannelGrid(config['nx'], config['ny'], config['lx'], config['ly'])
    reference = None
    if config['case'] in ('mms', 'discrete_mms'):
        reference = (DiscreteManufactured if config['case'] == 'discrete_mms' else Manufactured)(
            grid, config['nu'], config['amplitude'])
        force = reference.force
        initial = reference.amplitude(0.)*reference.initial_shape
    else:
        load = grid.pack_load(np.full((grid.ny, grid.nx+1), config['driving']),
                              np.zeros((grid.ny+1, grid.nx)))
        force = lambda t: load
        initial = np.zeros(grid.size) if config['case'] == 'startup' else \
            poiseuille(grid, config['nu'], config['driving'], discrete=True)
        if config['case'] == 'perturbed':
            initial += config['epsilon']*Manufactured(grid, config['nu']).initial_shape
        elif config['case'] == 'ts_wave':
            from .ts_mode import discrete_mode
            mode = discrete_mode(grid, config['nu'], config['driving'])
            initial += config['epsilon']*mode.centreline_speed*mode.velocity
    model = ChannelModel(grid, config['nu'], force=force, cache_size=config['cache_size'],
                         tolerance=config['stokes_tolerance'], backend=config['backend'])
    return model, model.state(0., initial, config['r0']), reference

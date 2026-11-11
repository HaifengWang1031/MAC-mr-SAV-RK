"""Exact paper sources plus explicitly chosen reproducible symmetry-breaking seed."""
import numpy as np

from solver.fourier import FourierSpace, MHDState
from solver.fourier.forced_mhd import ForcedFourierMHD
from solver.fourier.space import ComplexArray


def paper_force(space: FourierSpace, kf: int, f0: float, mu: float) -> ComplexArray:
    """curl f_u=2*f0*kf*cos(kf*x)cos(kf*y), g_a=mu*f0/kf*sin(kf*x)sin(kf*y)."""
    x, y = kf*space.x, kf*space.y
    u = np.array([-np.cos(x)*np.sin(y), np.sin(x)*np.cos(y)])
    b = np.array([np.sin(x)*np.cos(y), -np.cos(x)*np.sin(y)])
    return space.to_spectral(f0*np.array([u, mu*b]))


def initial_state(model: ForcedFourierMHD, kf: int, f0: float, mu: float,
                  seed: int, relative_rms: float) -> MHDState:
    """Grid-independent random low-mode stream/flux functions, not paper-specified ICs.

    Each vector RMS is relative_rms*sqrt(f0/kf), B has an extra factor mu.
    At mu=0 the run is exactly hydrodynamic. a and psi have no zero mode.
    """
    s = model.space
    rng = np.random.default_rng(seed)
    band = kf+1
    modes = np.arange(-band, band+1)
    ey = np.exp(1j*s.y[:, :1]*modes)
    ex = np.exp(1j*modes[:, None]*s.x[:1, :])
    vectors = []
    for factor in (1., mu):
        coefficients = rng.normal(size=(len(modes), len(modes)))+1j*rng.normal(size=(len(modes), len(modes)))
        potential = s.to_spectral((ey@coefficients@ex).real)
        potential[0, 0] = 0.
        vector = np.stack((1j*s.ky*potential, -1j*s.kx*potential))
        vector *= factor*relative_rms*np.sqrt(f0/kf)/np.sqrt(s.inner(vector, vector)/s.area)
        vectors.append(vector)
    return model.state(0., np.array(vectors))

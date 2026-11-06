"""Zhou et al. (2019) magnetic-island initial data and snapshot geometry."""
import numpy as np

from solver.fourier import FourierMHD, FourierSpace, MHDState
from solver.mac.grid import Array

ZHOU_CASE = 'zhou_2019'


def zhou_initial(model: FourierMHD, seed: int, relative_rms: float) -> MHDState:
    """Equation in Numerical study, with reproducible low-mode magnetic noise."""
    s = model.space
    flux = s.to_spectral(np.cos(8*s.x)*np.cos(8*s.y)/8)
    # Use grid-independent coefficients in a fixed low band.
    rng = np.random.default_rng(seed)
    modes = np.arange(-4, 5)
    coefficients = rng.normal(size=(9, 9))+1j*rng.normal(size=(9, 9))
    coefficients[4, 4] = 0.
    y_modes = np.exp(1j*s.y[:, :1]*modes)
    x_modes = np.exp(1j*modes[:, None]*s.x[:1, :])
    noise = s.to_spectral((y_modes@coefficients@x_modes).real)
    noise[0, 0] = 0.
    noise_rms = np.sqrt(s.inner(noise, s.k2*noise))
    flux += relative_rms*np.sqrt(s.inner(flux, s.k2*flux))/noise_rms*noise
    b = np.stack((1j*s.ky*flux, -1j*s.kx*flux))
    return model.state(0., np.stack((np.zeros_like(b), b)))


def magnetic_geometry(space: FourierSpace, magnetic: Array) -> tuple[Array, Array]:
    """Return zero-mean magnetic flux A and j = -ΔA from the saved B."""
    bx, by = space.to_spectral(magnetic)
    j = 1j*(space.kx*by-space.ky*bx)
    return space.to_physical(space.inverse_k2*j), space.to_physical(j)

"""Collocated doubly periodic space, forward-normalized real FFT and strict 2/3 band.

Physical arrays end in (ny,nx); spectral arrays end in (ny,nx//2+1).
Velocity has a component axis immediately before these two axes. There are no
duplicated periodic endpoints. Public transformations always allocate new arrays.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft2, rfft2

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class FourierSpace:
    nx: int
    ny: int
    lx: float = 2*np.pi
    ly: float = 2*np.pi
    x: RealArray = field(init=False, repr=False)
    y: RealArray = field(init=False, repr=False)
    kx: RealArray = field(init=False, repr=False)
    ky: RealArray = field(init=False, repr=False)
    k2: RealArray = field(init=False, repr=False)
    inverse_k2: RealArray = field(init=False, repr=False)
    weights: RealArray = field(init=False, repr=False)
    mask: NDArray[np.bool_] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.nx) is not int or type(self.ny) is not int or min(self.nx, self.ny) < 4:
            raise ValueError('Fourier grid dimensions must be integers >= 4')
        if not np.isfinite([self.lx, self.ly]).all() or min(self.lx, self.ly) <= 0:
            raise ValueError('Domain lengths must be finite and positive')
        x, y = np.meshgrid(np.arange(self.nx)*self.lx/self.nx,
                           np.arange(self.ny)*self.ly/self.ny)
        mx = np.arange(self.nx//2+1, dtype=float)[None, :]
        my = np.rint(np.fft.fftfreq(self.ny)*self.ny)[:, None]
        kx = np.broadcast_to(2*np.pi*mx/self.lx, self.spectral_shape).copy()
        ky = np.broadcast_to(2*np.pi*my/self.ly, self.spectral_shape).copy()
        k2 = kx*kx+ky*ky
        inverse = np.zeros_like(k2)
        np.divide(1., k2, out=inverse, where=k2 > 0)
        weights = np.full(self.spectral_shape, 2.)
        weights[:, 0] = 1.
        if self.nx % 2 == 0:
            weights[:, -1] = 1.
        mask = (mx < self.nx/3) & (np.abs(my) < self.ny/3)
        for name, value in (('x', x), ('y', y), ('kx', kx), ('ky', ky),
                            ('k2', k2), ('inverse_k2', inverse),
                            ('weights', weights), ('mask', mask)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def spectral_shape(self) -> tuple[int, int]:
        return self.ny, self.nx//2+1

    @property
    def area(self) -> float:
        return self.lx*self.ly

    def metadata(self) -> dict[str, Any]:
        xmax = int(np.ceil(self.nx/3)-1)
        ymax = int(np.ceil(self.ny/3)-1)
        return {'boundary': 'periodic_xy', 'discretization': 'fourier_leray',
                'field_layout': 'collocated_yx', 'fft_norm': 'forward',
                'dealias': 'strict_2_3', 'retained_mx': [-xmax, xmax],
                'retained_my': [-ymax, ymax],
                'retained_full_modes': (2*xmax+1)*(2*ymax+1),
                'retained_half_modes': int(np.count_nonzero(self.mask))}

    def to_spectral(self, values: RealArray, *, filter: bool = True) -> ComplexArray:
        array = np.asarray(values)
        if array.shape[-2:] != self.shape or np.iscomplexobj(array) or not np.isfinite(array).all():
            raise ValueError('Expected finite real periodic samples ending in (ny,nx)')
        transformed: ComplexArray = rfft2(array, axes=(-2, -1), norm='forward', workers=1)
        return transformed*self.mask if filter else transformed

    def to_physical(self, coefficients: ComplexArray) -> RealArray:
        array = np.asarray(coefficients)
        if array.shape[-2:] != self.spectral_shape or not np.isfinite(array).all():
            raise ValueError('Invalid Fourier coefficients')
        return irfft2(array, s=self.shape, axes=(-2, -1), norm='forward', workers=1)

    def velocity(self, u: RealArray, v: RealArray, *, project: bool = True) -> ComplexArray:
        result = self.to_spectral(np.stack([u, v]))
        return self.project(result) if project else result

    def check_velocity(self, coefficients: ComplexArray) -> ComplexArray:
        """Validate the retained real-field representation, without silently truncating."""
        array = np.asarray(coefficients, dtype=complex)
        if array.shape != (2, *self.spectral_shape) or not np.isfinite(array).all():
            raise ValueError('Expected finite spectral velocity of shape (2,ny,nx//2+1)')
        if np.any(array[..., ~self.mask] != 0):
            raise ValueError('Velocity/load contains modes outside the strict 2/3 band')
        # On kx=0 the y coefficients must pair with their complex conjugates.
        mirror = (-np.arange(self.ny)) % self.ny
        discrepancy = array[:, :, 0]-np.conj(array[:, mirror, 0])
        if np.max(np.abs(discrepancy)) > 1e-12*(1+np.max(np.abs(array))):
            raise ValueError('Coefficients do not represent a real periodic velocity')
        return array

    def project(self, velocity: ComplexArray) -> ComplexArray:
        dot = self.kx*velocity[..., 0, :, :]+self.ky*velocity[..., 1, :, :]
        return velocity-self.gradient(-1j*dot*self.inverse_k2)

    def gradient(self, pressure: ComplexArray) -> ComplexArray:
        return np.stack([1j*self.kx*pressure, 1j*self.ky*pressure], axis=-3)

    def divergence(self, velocity: ComplexArray) -> ComplexArray:
        return 1j*(self.kx*velocity[..., 0, :, :]+self.ky*velocity[..., 1, :, :])

    def inner(self, left: ComplexArray, right: ComplexArray) -> float:
        """Real L2 pairing, including conjugation, half-spectrum multiplicity and area."""
        return float(self.area*np.sum(self.weights*np.real(np.conj(left)*right)))

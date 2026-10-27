"""FFT acceleration of the SAME periodic MAC saddle system, not spectral derivatives."""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.fft import irfft, rfft
from scipy.linalg.lapack import zgbtrf, zgbtrs

from ..mac.grid import Array
from ..mac.operators import _laplacian_1d
from ..mac.stokes import StokesResult
from .operators import ChannelOperators


@dataclass
class _Band:
    lu: np.ndarray
    pivots: np.ndarray
    bandwidth: int

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        result, info = zgbtrs(self.lu, self.bandwidth, self.bandwidth,
                             np.asfortranarray(rhs, dtype=complex), self.pivots)
        if info != 0: raise RuntimeError(f'Channel band solve info={info}')
        return result


def _factor(matrix: sp.spmatrix, bandwidth: int) -> _Band:
    coo = matrix.tocoo()
    if np.max(np.abs(coo.row-coo.col), initial=0) > bandwidth:
        raise RuntimeError('Channel matrix exceeds declared bandwidth')
    # LAPACK reserves bandwidth extra rows above the original upper band for LU fill.
    ab = np.zeros((3*bandwidth+1, matrix.shape[0]), dtype=complex, order='F')
    ab[2*bandwidth+coo.row-coo.col, coo.col] = coo.data
    lu, pivots, info = zgbtrf(ab, bandwidth, bandwidth, overwrite_ab=True)
    if info != 0: raise RuntimeError(f'Channel band factorization info={info}')
    return _Band(lu, pivots, bandwidth)


class FFTStokes:
    """One banded velocity-pressure solve per x mode, with bounded shift-factor reuse.

    The staggered x divergence symbol is (exp(i*theta)-1)/hx; gradient is its
    negative conjugate and the Laplacian eigenvalue is 4*sin(theta/2)^2/hx².
    Nonzero modes retain all pressure rows. Only the zero-mode pressure mean is gauged.
    """
    def __init__(self, operators: ChannelOperators, cache_size: int = 4,
                 tolerance: float = 1e-10) -> None:
        if type(cache_size) is not int or cache_size < 1 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Invalid channel FFT solver controls')
        self.ops, self.cache_size, self.tolerance = operators, cache_size, tolerance
        self.cache: OrderedDict[tuple[float, float], tuple[_Band, list[_Band]]] = OrderedDict()
        self.factorizations = 0
        self.factorization_seconds = 0.
        self.solve_seconds = 0.
        grid = operators.grid
        self._ku = _laplacian_1d(grid.ny, grid.hy, True)
        self._kv = _laplacian_1d(grid.ny-1, grid.hy, False)
        self._dy = sp.diags([np.ones(grid.ny-1), -np.ones(grid.ny-1)], [0, -1],
                           shape=(grid.ny, grid.ny-1), format='csr')/grid.hy
        order = []
        for j in range(grid.ny):
            order.extend([j, 2*grid.ny-1+j])
            if j < grid.ny-1: order.append(grid.ny+j)
        self._order = np.array(order)

    def _factors(self, mass: float, viscosity: float) -> tuple[_Band, list[_Band]]:
        key = (float(mass), float(viscosity))
        if key not in self.cache:
            start = perf_counter()
            grid = self.ops.grid
            iu, iv = sp.eye(grid.ny, format='csr'), sp.eye(grid.ny-1, format='csr')
            hu0 = mass*iu+viscosity*self._ku
            zero = _factor(hu0, 1)
            modes = []
            for m in range(1, grid.nx//2+1):
                theta = 2*np.pi*m/grid.nx
                d = (np.exp(1j*theta)-1)/grid.hx
                if grid.nx % 2 == 0 and m == grid.nx//2: d = complex(-2/grid.hx)
                eigenvalue = 4*np.sin(theta/2)**2/grid.hx**2
                hu = hu0+viscosity*eigenvalue*iu
                hv = mass*iv+viscosity*(self._kv+eigenvalue*iv)
                # Negated continuity makes the off-diagonal blocks adjoints.
                matrix = sp.bmat([[hu, None, -np.conj(d)*iu],
                                  [None, hv, -self._dy.T],
                                  [-d*iu, -self._dy, None]], format='csr')
                modes.append(_factor(matrix[self._order, :][:, self._order], 3))
            self.cache[key] = zero, modes
            self.factorizations += 1
            self.factorization_seconds += perf_counter()-start
            if len(self.cache) > self.cache_size: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> StokesResult:
        grid = self.ops.grid
        if not np.isfinite([mass, viscosity]).all() or min(mass, viscosity) < 0 or mass+viscosity <= 0:
            raise ValueError('Require nonnegative mass/viscosity with nonzero total')
        if rhs.ndim not in (1, 2) or rhs.shape[0] != grid.size \
                or np.iscomplexobj(rhs) or not np.isfinite(rhs).all():
            raise ValueError('Invalid channel FFT RHS')
        zero, factors = self._factors(mass, viscosity)
        start = perf_counter()
        force = rhs if rhs.ndim == 2 else rhs[:, None]
        columns = force.shape[1]
        fu = rfft(force[:grid.nu].reshape(grid.ny, grid.nx, columns), axis=1, norm='ortho', workers=1)
        fv = rfft(force[grid.nu:].reshape(grid.ny-1, grid.nx, columns), axis=1, norm='ortho', workers=1)
        loads = np.concatenate([fu, fv], axis=0)
        velocity_hat = np.zeros_like(loads)
        pressure_hat = np.zeros((grid.ny, grid.nx//2+1, columns), dtype=complex)
        velocity_hat[:grid.ny, 0] = zero.solve(fu[:, 0])
        # Dy v=0 and v at both walls=0 imply v_0=0; G_y p_0=fv_0.
        pressure_hat[1:, 0] = grid.hy*np.cumsum(fv[:, 0], axis=0)
        pressure_hat[:, 0] -= pressure_hat[:, 0].mean(axis=0)
        for m, factor in enumerate(factors, start=1):
            packed = np.vstack([loads[:, m], np.zeros((grid.ny, columns))])
            solved = factor.solve(packed[self._order])
            original = np.empty_like(solved)
            original[self._order] = solved
            velocity_hat[:, m] = original[:2*grid.ny-1]
            pressure_hat[:, m] = original[2*grid.ny-1:]
        velocity = irfft(velocity_hat, n=grid.nx, axis=1, norm='ortho', workers=1).reshape(grid.size, columns)
        pressure = irfft(pressure_hat, n=grid.nx, axis=1, norm='ortho', workers=1)
        pressure -= pressure.mean(axis=(0, 1), keepdims=True)
        self.solve_seconds += perf_counter()-start
        defect = mass*velocity+viscosity*(self.ops.K@velocity) \
            +self.ops.G@pressure.reshape(grid.np, columns)-force
        residual = float(np.max(np.abs(defect))/(1+np.max(np.abs(force))))
        divergence = float(np.max(np.abs(self.ops.D@velocity)))
        if not np.isfinite(velocity).all() or not np.isfinite(pressure).all() \
                or residual > self.tolerance or divergence > self.tolerance*(1+np.max(np.abs(velocity))):
            raise RuntimeError(f'Channel FFT residual={residual:.3e}, divergence={divergence:.3e}')
        if rhs.ndim == 1:
            return StokesResult(velocity[:, 0], pressure[:, :, 0], residual, divergence)
        return StokesResult(velocity, pressure, residual, divergence)

"""Shift-reusable MAC Stokes solve by tensor diagonalization and Schur CG."""
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.fft import dct, dst, idct, idst
from scipy.sparse.linalg import LinearOperator, cg

from .grid import Array
from .operators import MACOperators
from .stokes import StokesResult


@dataclass(frozen=True)
class _TensorBasis:
    transform_y: "_Transform"
    transform_x: "_Transform"
    eigenvalues: Array


_Transform = Literal["dct2", "dst1", "dst2"]


def _eigenvalues_1d(n: int, spacing: float, transform: _Transform) -> Array:
    if transform == "dct2":
        angles = np.pi * np.arange(n) / n
    elif transform == "dst1":
        angles = np.pi * np.arange(1, n + 1) / (n + 1)
    else:
        angles = np.pi * np.arange(1, n + 1) / n
    return (2.0 - 2.0 * np.cos(angles)) / spacing**2


def _tensor_basis(
    shape: tuple[int, int],
    spacings: tuple[float, float],
    transforms: tuple[_Transform, _Transform],
) -> _TensorBasis:
    ny, nx = shape
    hy, hx = spacings
    transform_y, transform_x = transforms
    eigenvalues_y = _eigenvalues_1d(ny, hy, transform_y)
    eigenvalues_x = _eigenvalues_1d(nx, hx, transform_x)
    return _TensorBasis(
        transform_y,
        transform_x,
        eigenvalues_y[:, None] + eigenvalues_x[None, :],
    )


def _transform_axis(values: Array, transform: _Transform, axis: int, inverse: bool) -> Array:
    if transform == "dct2":
        function = idct if inverse else dct
        return function(values, type=2, axis=axis, norm="ortho")
    function = idst if inverse else dst
    transform_type = 1 if transform == "dst1" else 2
    return function(values, type=transform_type, axis=axis, norm="ortho")


def _transform(values: Array, basis: _TensorBasis, inverse: bool) -> Array:
    transformed = _transform_axis(values, basis.transform_y, 0, inverse)
    return _transform_axis(transformed, basis.transform_x, 1, inverse)


def _apply_inverse(values: Array, basis: _TensorBasis, shift: Array) -> Array:
    transformed = _transform(values, basis, inverse=False)
    transformed /= shift[:, :, None]
    return _transform(transformed, basis, inverse=True)


class TensorStokes:
    """No-slip MAC Stokes backend that reuses fixed tensor eigenspaces for all shifts.

    The velocity Helmholtz blocks are inverted exactly by tensor DST/DCT transforms.
    The pressure Schur complement is applied exactly and solved by preconditioned
    conjugate gradients on the mean-zero pressure space.  The preconditioner is
    ``mass * (D D.T)^dagger + viscosity * I``; only the diagonal spectral shifts
    change when the time step changes.
    """

    def __init__(
        self,
        operators: MACOperators,
        tolerance: float = 1e-9,
        krylov_tolerance: float = 1e-12,
        max_iterations: int = 100,
    ) -> None:
        if (
            tolerance <= 0
            or krylov_tolerance <= 0
            or not np.isfinite([tolerance, krylov_tolerance]).all()
            or max_iterations < 1
        ):
            raise ValueError("Invalid solver controls")
        start = perf_counter()
        self.ops = operators
        self.tolerance = tolerance
        self.krylov_tolerance = krylov_tolerance
        self.max_iterations = max_iterations
        grid = operators.grid
        self._u_basis = _tensor_basis(
            (grid.ny, grid.nx - 1), (grid.hy, grid.hx), ("dst2", "dst1")
        )
        self._v_basis = _tensor_basis(
            (grid.ny - 1, grid.nx), (grid.hy, grid.hx), ("dst1", "dst2")
        )
        self._pressure_basis = _tensor_basis(
            (grid.ny, grid.nx), (grid.hy, grid.hx), ("dct2", "dct2")
        )
        pressure_scale = float(np.max(self._pressure_basis.eigenvalues))
        self._pressure_positive = self._pressure_basis.eigenvalues > (
            100.0 * np.finfo(float).eps * pressure_scale
        )
        self.setup_seconds = perf_counter() - start
        self.solve_seconds = 0.0
        self.solves = 0
        self.right_hand_sides = 0
        self.total_iterations = 0
        self.last_iterations: tuple[int, ...] = ()

    def _velocity_inverse(self, values: Array, mass: float, viscosity: float) -> Array:
        grid = self.ops.grid
        columns = values.shape[1]
        u_values = values[: grid.nu].reshape(grid.ny, grid.nx - 1, columns)
        v_values = values[grid.nu :].reshape(grid.ny - 1, grid.nx, columns)
        u_shift = mass + viscosity * self._u_basis.eigenvalues
        v_shift = mass + viscosity * self._v_basis.eigenvalues
        u_solution = _apply_inverse(u_values, self._u_basis, u_shift)
        v_solution = _apply_inverse(v_values, self._v_basis, v_shift)
        return np.vstack(
            [u_solution.reshape(grid.nu, columns), v_solution.reshape(grid.nv, columns)]
        )

    def _pressure_poisson_inverse(self, values: Array) -> Array:
        grid = self.ops.grid
        pressure = np.asarray(values, dtype=float).reshape(grid.ny, grid.nx)
        pressure = pressure - pressure.mean()
        transformed = _transform(pressure, self._pressure_basis, inverse=False)
        transformed = np.divide(
            transformed,
            self._pressure_basis.eigenvalues,
            out=np.zeros_like(transformed),
            where=self._pressure_positive,
        )
        solution = _transform(transformed, self._pressure_basis, inverse=True)
        return (solution - solution.mean()).reshape(-1)

    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> StokesResult:
        if not np.isfinite([mass, viscosity]).all() or min(mass, viscosity) < 0 or mass + viscosity <= 0:
            raise ValueError("Require mass,viscosity >= 0 and nonzero total")
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.ops.grid.size or not np.isfinite(rhs).all():
            raise ValueError("Invalid Stokes RHS")

        start = perf_counter()
        two_dimensional = rhs.ndim == 2
        force = np.asarray(rhs, dtype=float) if two_dimensional else np.asarray(rhs, dtype=float)[:, None]
        inverse_force = self._velocity_inverse(force, mass, viscosity)
        pressure_rhs = -(self.ops.D @ inverse_force)
        pressure_rhs -= pressure_rhs.mean(axis=0)

        def schur_action(pressure: Array) -> Array:
            mean_zero = pressure - pressure.mean()
            lifted = np.asarray(self.ops.D.T @ mean_zero).reshape(-1, 1)
            result = self.ops.D @ self._velocity_inverse(lifted, mass, viscosity)
            result -= result.mean()
            return np.asarray(result).reshape(-1)

        def precondition(pressure: Array) -> Array:
            mean_zero = pressure - pressure.mean()
            result = mass * self._pressure_poisson_inverse(mean_zero) + viscosity * mean_zero
            return result - result.mean()

        pressure_size = self.ops.grid.np
        schur = LinearOperator((pressure_size, pressure_size), matvec=schur_action, dtype=float)
        preconditioner = LinearOperator(
            (pressure_size, pressure_size), matvec=precondition, dtype=float
        )
        pressures = np.empty_like(pressure_rhs)
        iterations: list[int] = []
        for column in range(force.shape[1]):
            iteration_count = 0

            def count_iteration(_: Array) -> None:
                nonlocal iteration_count
                iteration_count += 1

            if np.max(np.abs(pressure_rhs[:, column])) == 0.0:
                pressure = np.zeros(pressure_size)
                info = 0
            else:
                pressure, info = cg(
                    schur,
                    pressure_rhs[:, column],
                    M=preconditioner,
                    rtol=self.krylov_tolerance,
                    atol=0.0,
                    maxiter=self.max_iterations,
                    callback=count_iteration,
                )
            if info != 0:
                raise RuntimeError(f"Pressure Schur CG failed with info={info}")
            pressures[:, column] = pressure - pressure.mean()
            iterations.append(iteration_count)

        velocity_rhs = force + self.ops.D.T @ pressures
        velocity = self._velocity_inverse(np.asarray(velocity_rhs), mass, viscosity)
        defect = (
            mass * velocity
            + viscosity * (self.ops.K @ velocity)
            + self.ops.G @ pressures
            - force
        )
        residual = float(np.max(np.abs(defect)) / (1 + np.max(np.abs(force))))
        divergence = float(np.max(np.abs(self.ops.D @ velocity)))
        self.solve_seconds += perf_counter() - start
        self.solves += 1
        self.right_hand_sides += force.shape[1]
        self.total_iterations += sum(iterations)
        self.last_iterations = tuple(iterations)
        if (
            not np.isfinite(velocity).all()
            or not np.isfinite(pressures).all()
            or residual > self.tolerance
            or divergence > self.tolerance * (1 + np.max(np.abs(velocity)))
        ):
            raise RuntimeError(f"Stokes residual={residual:.3e}, divergence={divergence:.3e}")
        if not two_dimensional:
            return StokesResult(
                velocity[:, 0],
                pressures[:, 0].reshape(self.ops.grid.ny, self.ops.grid.nx),
                residual,
                divergence,
            )
        return StokesResult(
            velocity,
            pressures.reshape(self.ops.grid.ny, self.ops.grid.nx, -1),
            residual,
            divergence,
        )

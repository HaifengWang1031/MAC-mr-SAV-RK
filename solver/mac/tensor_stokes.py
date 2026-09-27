"""Shift-reusable MAC Stokes solve by tensor diagonalization and Schur CG."""
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.fft import dct, dst, idct, idst
from scipy.sparse.linalg import LinearOperator, cg

from .grid import Array
from .kernels import (
    packed_divergence,
    packed_divergence_transpose,
    packed_divergence_transpose_vector,
    packed_divergence_vector,
)
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


def _transform_axis(
    values: Array,
    transform: _Transform,
    axis: int,
    inverse: bool,
    overwrite_input: bool,
) -> Array:
    if transform == "dct2":
        function = idct if inverse else dct
        return function(
            values,
            type=2,
            axis=axis,
            norm="ortho",
            overwrite_x=overwrite_input,
        )
    function = idst if inverse else dst
    transform_type = 1 if transform == "dst1" else 2
    return function(
        values,
        type=transform_type,
        axis=axis,
        norm="ortho",
        overwrite_x=overwrite_input,
    )


def _transform(
    values: Array, basis: _TensorBasis, inverse: bool, reuse_buffers: bool
) -> Array:
    transformed = _transform_axis(
        values,
        basis.transform_y,
        0,
        inverse,
        overwrite_input=reuse_buffers and inverse,
    )
    return _transform_axis(
        transformed,
        basis.transform_x,
        1,
        inverse,
        overwrite_input=reuse_buffers,
    )


def _apply_inverse(
    values: Array, basis: _TensorBasis, shift: Array, reuse_buffers: bool
) -> Array:
    transformed = _transform(
        values, basis, inverse=False, reuse_buffers=reuse_buffers
    )
    transformed /= shift[:, :, None]
    return _transform(transformed, basis, inverse=True, reuse_buffers=reuse_buffers)


def _column_means(values: Array) -> Array:
    return np.asarray([np.mean(values[:, column]) for column in range(values.shape[1])])


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
        matrix_free_operators: bool = True,
        reuse_transform_buffers: bool = True,
    ) -> None:
        if (
            tolerance <= 0
            or krylov_tolerance <= 0
            or not np.isfinite([tolerance, krylov_tolerance]).all()
            or max_iterations < 1
        ):
            raise ValueError("Invalid solver controls")
        if not isinstance(matrix_free_operators, bool):
            raise ValueError("matrix_free_operators must be a bool")
        if not isinstance(reuse_transform_buffers, bool):
            raise ValueError("reuse_transform_buffers must be a bool")
        start = perf_counter()
        self.ops = operators
        self.tolerance = tolerance
        self.krylov_tolerance = krylov_tolerance
        self.max_iterations = max_iterations
        self.matrix_free_operators = matrix_free_operators
        self.reuse_transform_buffers = reuse_transform_buffers
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

    def _divergence(self, velocity: Array) -> Array:
        if not self.matrix_free_operators:
            return np.asarray(self.ops.D @ velocity)
        grid = self.ops.grid
        if velocity.shape[1] == 1:
            result = packed_divergence_vector(
                velocity[:, 0], grid.nx, grid.ny, grid.hx, grid.hy
            )
            return result[:, None]
        return packed_divergence(velocity, grid.nx, grid.ny, grid.hx, grid.hy)

    def _divergence_transpose(self, pressure: Array) -> Array:
        if not self.matrix_free_operators:
            return np.asarray(self.ops.D.T @ pressure)
        grid = self.ops.grid
        if pressure.shape[1] == 1:
            result = packed_divergence_transpose_vector(
                pressure[:, 0], grid.nx, grid.ny, grid.hx, grid.hy
            )
            return result[:, None]
        return packed_divergence_transpose(
            pressure, grid.nx, grid.ny, grid.hx, grid.hy
        )

    def _velocity_inverse(
        self, values: Array, u_shift: Array, v_shift: Array
    ) -> Array:
        grid = self.ops.grid
        columns = values.shape[1]
        u_values = values[: grid.nu].reshape(grid.ny, grid.nx - 1, columns)
        v_values = values[grid.nu :].reshape(grid.ny - 1, grid.nx, columns)
        u_solution = _apply_inverse(
            u_values, self._u_basis, u_shift, self.reuse_transform_buffers
        )
        v_solution = _apply_inverse(
            v_values, self._v_basis, v_shift, self.reuse_transform_buffers
        )
        return np.vstack(
            [u_solution.reshape(grid.nu, columns), v_solution.reshape(grid.nv, columns)]
        )

    def _pressure_poisson_inverse(self, values: Array) -> Array:
        grid = self.ops.grid
        vector_input = values.ndim == 1
        columns = np.asarray(values, dtype=float).reshape(grid.np, -1)
        columns = columns - _column_means(columns)
        pressure = columns.reshape(grid.ny, grid.nx, -1)
        transformed = _transform(
            pressure,
            self._pressure_basis,
            inverse=False,
            reuse_buffers=self.reuse_transform_buffers,
        )
        transformed = np.divide(
            transformed,
            self._pressure_basis.eigenvalues[:, :, None],
            out=np.zeros_like(transformed),
            where=self._pressure_positive[:, :, None],
        )
        solution = _transform(
            transformed,
            self._pressure_basis,
            inverse=True,
            reuse_buffers=self.reuse_transform_buffers,
        )
        result = solution.reshape(grid.np, -1)
        result -= _column_means(result)
        return result[:, 0] if vector_input else result

    def solve(self, rhs: Array, *, mass: float, viscosity: float) -> StokesResult:
        if (
            not np.isfinite([mass, viscosity]).all()
            or min(mass, viscosity) < 0
            or mass + viscosity <= 0
        ):
            raise ValueError("Require mass,viscosity >= 0 and nonzero total")
        if (
            rhs.ndim not in (1, 2)
            or rhs.shape[0] != self.ops.grid.size
            or not np.isfinite(rhs).all()
        ):
            raise ValueError("Invalid Stokes RHS")

        start = perf_counter()
        two_dimensional = rhs.ndim == 2
        force = (
            np.asarray(rhs, dtype=float)
            if two_dimensional
            else np.asarray(rhs, dtype=float)[:, None]
        )
        u_shift = mass + viscosity * self._u_basis.eigenvalues
        v_shift = mass + viscosity * self._v_basis.eigenvalues
        inverse_force = self._velocity_inverse(force, u_shift, v_shift)
        pressure_rhs = -self._divergence(inverse_force)
        pressure_rhs -= pressure_rhs.mean(axis=0)

        def schur_action(pressure: Array) -> Array:
            mean_zero = pressure - _column_means(pressure)
            lifted = self._divergence_transpose(mean_zero)
            result = self._divergence(
                self._velocity_inverse(lifted, u_shift, v_shift)
            )
            result -= _column_means(result)
            return np.asarray(result)

        def precondition(pressure: Array) -> Array:
            mean_zero = pressure - _column_means(pressure)
            result = (
                mass * self._pressure_poisson_inverse(mean_zero)
                + viscosity * mean_zero
            )
            return result - _column_means(result)

        pressure_size = self.ops.grid.np
        schur = LinearOperator(
            (pressure_size, pressure_size),
            matvec=lambda pressure: schur_action(pressure[:, None])[:, 0],
            dtype=float,
        )
        preconditioner = LinearOperator(
            (pressure_size, pressure_size),
            matvec=lambda pressure: precondition(pressure[:, None])[:, 0],
            dtype=float,
        )
        pressures = np.empty_like(pressure_rhs)
        iterations = []
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

        pressures -= _column_means(pressures)

        pressure_lift = self._divergence_transpose(pressures)
        velocity_rhs = force + pressure_lift
        velocity = self._velocity_inverse(np.asarray(velocity_rhs), u_shift, v_shift)
        defect = (
            mass * velocity
            + viscosity * (self.ops.K @ velocity)
            - pressure_lift
            - force
        )
        residual = float(np.max(np.abs(defect)) / (1 + np.max(np.abs(force))))
        divergence = float(np.max(np.abs(self._divergence(velocity))))
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

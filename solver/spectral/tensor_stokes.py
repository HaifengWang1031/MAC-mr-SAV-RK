"""Optional Legendre fast diagonalization with the true pressure Schur action.

One-dimensional generalized eigenspaces are independent of stage coefficients.
The pressure mass-Schur/mass combination is a preconditioner only; CG acts on
B H^{-1} B^T, and SpectralStokes retains all per-column equation checks.
"""
from typing import Any

import numpy as np
from scipy.linalg import eigh, cho_solve_banded
from scipy.sparse.linalg import LinearOperator, cg

from . import basis
from .assembly import Array, Space
from .mass import TensorMass
from .stokes import SpectralStokes


class _TensorInverse:
    def __init__(self, space: Space) -> None:
        self.size = space.size
        mass = basis.mass_reference(self.size)
        stiffness = basis.stiffness_reference(self.size)
        self.lx, self.qx = eigh(stiffness*basis.stiffness_scale(space.lx),
                                mass*basis.mass_scale(space.lx))
        self.ly, self.qy = eigh(stiffness*basis.stiffness_scale(space.ly),
                                mass*basis.mass_scale(space.ly))
        self.eigenvalues = self.lx[:, None]+self.ly[None, :]
        self.pressure = False

    def apply(self, load: Array, mass: float, viscosity: float) -> Array:
        values = np.asarray(load).reshape(self.size, self.size)
        transformed = self.qx.T @ values @ self.qy
        denominator = mass+viscosity*self.eigenvalues
        if self.pressure and mass == 0:
            denominator = denominator.copy()
            denominator[0, 0] = 1.
            transformed[0, 0] = 0.
        result = self.qx @ (transformed/denominator) @ self.qy.T
        if self.pressure and mass == 0:
            result[0, 0] = 0.
        return result.reshape(-1)


class _MassSchurInverse(_TensorInverse):
    """Exact inverse of B Mv^{-1} B^T on the mean-zero pressure space.

    For D=D1 and P=pressure/velocity cross mass, the two 1D pairs are
    A=D M^{-1} D^T, C=P M^{-1} P^T. The full operator is
    Ax kron Cy + Cx kron Ay. Only the tensor constant mode is singular.
    """
    def __init__(self, space: Space) -> None:
        self.size, self.pressure = space.size, True
        factors = TensorMass(space)
        d = basis.divergence_reference(self.size)
        eigenvalues, eigenvectors = [], []
        for length, factor in ((space.lx, factors.x), (space.ly, factors.y)):
            p = basis.projection_reference(self.size)*basis.mass_scale(length)
            a = d @ cho_solve_banded((factor, True), d.T)
            c = p @ cho_solve_banded((factor, True), p.T)
            values, vectors = eigh((a+a.T)*.5, (c+c.T)*.5)
            values[0] = 0.
            eigenvalues.append(values)
            eigenvectors.append(vectors)
        self.lx, self.ly = eigenvalues
        self.qx, self.qy = eigenvectors
        self.eigenvalues = self.lx[:, None]+self.ly[None, :]

class _ShiftedFactor:
    def __init__(self, backend: "TensorStokes", mass: float, viscosity: float) -> None:
        self.backend, self.mass, self.viscosity = backend, mass, viscosity

    def solve(self, loads: Array) -> Array:
        backend, mass, viscosity = self.backend, self.mass, self.viscosity
        size = backend.space.velocity_dofs
        dx, dy = backend.divergence_x[1:], backend.divergence_y[1:]
        def inverse(load: Array) -> Array:
            backend.helmholtz_applications += 1
            return backend.velocity_inverse.apply(load, mass, viscosity)
        def schur(pressure: Array) -> Array:
            return dx @ inverse(dx.T @ pressure)+dy @ inverse(dy.T @ pressure)
        def precondition(pressure: Array) -> Array:
            full = np.r_[0., pressure]
            out = viscosity*full/backend.pressure_mass
            if mass:
                out += mass*backend.pressure_inverse.apply(full, 0., 1.)
            return out[1:]
        operator = LinearOperator((size-1, size-1), matvec=schur, dtype=np.float64)
        preconditioner = LinearOperator(operator.shape, matvec=precondition, dtype=np.float64)
        solutions = np.empty_like(loads)
        for index in range(loads.shape[1]):
            u0, v0 = inverse(loads[:size, index]), inverse(loads[size:2*size, index])
            rhs = loads[2*size:, index]-dx @ u0-dy @ v0
            iterations = 0
            def count(_pressure: Array) -> None:
                nonlocal iterations
                iterations += 1
            if size > 1:
                pressure, info = cg(operator, rhs, M=preconditioner,
                                    rtol=backend.krylov_tolerance, atol=0.,
                                    maxiter=backend.max_iterations, callback=count)
                backend.total_iterations += iterations
                backend.schur_rhs += 1
                if info != 0 or not np.isfinite(pressure).all():
                    raise RuntimeError(f'Spectral Schur CG failed: info={info}, iterations={iterations}')
                u0 += inverse(dx.T @ pressure)
                v0 += inverse(dy.T @ pressure)
                solutions[2*size:, index] = pressure
            solutions[:size, index], solutions[size:2*size, index] = u0, v0
        return solutions


class TensorStokes(SpectralStokes):
    """Same weak Stokes equations; reusable tensor inverses and mean-zero pressure CG."""

    def __init__(self, space: Space, *, tolerance: float = 1e-9, cache_size: int = 4,
                 krylov_tolerance: float = 1e-12, max_iterations: int = 200) -> None:
        if not np.isfinite(krylov_tolerance) or krylov_tolerance <= 0 or max_iterations < 1:
            raise ValueError('Invalid spectral Krylov controls')
        super().__init__(space, tolerance=tolerance, cache_size=cache_size)
        self.velocity_inverse = _TensorInverse(space)
        self.pressure_inverse = _MassSchurInverse(space)
        mx = basis.legendre_mass_diagonal(space.size)*basis.mass_scale(space.lx)
        my = basis.legendre_mass_diagonal(space.size)*basis.mass_scale(space.ly)
        self.pressure_mass = np.outer(mx, my).reshape(-1)
        self.krylov_tolerance = krylov_tolerance
        self.max_iterations = max_iterations
        self.total_iterations = self.schur_rhs = self.helmholtz_applications = 0

    def _system(self, mass: float, viscosity: float) -> Any:
        key = (float(mass), float(viscosity))
        if key not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[key] = (_ShiftedFactor(self, mass, viscosity), None)
        self.cache.move_to_end(key)
        return self.cache[key]

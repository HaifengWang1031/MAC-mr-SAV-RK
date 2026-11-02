"""Optional real-float64 PETSc block FGMRES for the exact shifted Stokes system.

Only the preconditioner uses diagonal mass approximations. Original matrices,
all continuity rows and the weighted pressure gauge remain in the outer system.
The initial adapter uses COMM_SELF; it does not claim distributed FEM assembly.
"""
from collections.abc import Sequence
from time import perf_counter
from typing import Any

import numpy as np
import scipy.sparse as sp

from .space import Array


def _petsc() -> Any:
    try:
        from petsc4py import PETSc
    except ImportError as error:
        raise RuntimeError('The PETSc FEM backend requires matching petsc4py/PETSc libraries') from error
    if np.dtype(PETSc.ScalarType) != np.dtype('float64'):
        raise RuntimeError('The FEM backend requires a real float64 PETSc build')
    if PETSc.COMM_WORLD.getSize() != 1:
        raise RuntimeError('This FEM adapter uses COMM_SELF; distributed assembly is not implemented')
    return PETSc


def _matrix(PETSc: Any, values: Any) -> tuple[Any, Any]:
    csr = values.tocsr(copy=True)
    csr.sum_duplicates();csr.sort_indices()
    csr.indptr = np.asarray(csr.indptr, dtype=PETSc.IntType)
    csr.indices = np.asarray(csr.indices, dtype=PETSc.IntType)
    return PETSc.Mat().createAIJ(size=csr.shape,
        csr=(csr.indptr, csr.indices, csr.data), comm=PETSc.COMM_SELF), csr


class _AMGInverse:
    """A fixed number of cycles is a preconditioner, never an accepted solution."""
    def __init__(self, PETSc: Any, matrix: Any, kind: str, cycles: int = 1) -> None:
        self.matrix, self.csr = _matrix(PETSc, matrix)
        self.ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        self.ksp.setOperators(self.matrix)
        self.ksp.setType('richardson')
        self.ksp.setNormType(PETSc.KSP.NormType.NONE)
        self.ksp.setTolerances(rtol=0., atol=0., max_it=cycles)
        self.ksp.setErrorIfNotConverged(False)
        pc = self.ksp.getPC()
        pc.setType(kind)
        if kind == 'hypre':
            pc.setHYPREType('boomeramg')
        self.ksp.setUp()
        self.rhs, self.value = self.matrix.createVecRight(), self.matrix.createVecRight()

    def solve(self, rhs: Array) -> Array:
        self.rhs.getArray()[:] = rhs
        self.value.set(0.)
        self.ksp.solve(self.rhs, self.value)
        return self.value.getArray(readonly=True).copy()

    def close(self) -> None:
        self.ksp.destroy();self.rhs.destroy();self.value.destroy();self.matrix.destroy()


class _BlockPreconditioner:
    def __init__(self, PETSc: Any, mass: Any, stiffness: Any, divergence: Any,
                 weights: Array, gauge: bool, reaction: float,
                 components: Sequence[np.ndarray], kind: str) -> None:
        self.PETSc, self.mass, self.stiffness = PETSc, mass, stiffness
        self.B, self.weights, self.gauge = divergence, weights, gauge
        self.reaction, self.kind = reaction, kind
        self.nv, self.np = mass.shape[0], len(weights)
        self.scale = np.r_[1/np.asarray(mass.diagonal()), np.ones(self.np+int(gauge))]
        self.alpha, self.beta = 0., 0.
        self.anchor: tuple[float, float] | None = None
        self.velocity: _AMGInverse | None = None
        self.parts: tuple[np.ndarray, ...] = ()
        if len(components) == 2:
            a, b = components
            if len(a) == len(b) and len(a)+len(b) == self.nv \
                    and not mass[a][:, b].nnz and not stiffness[a][:, b].nnz \
                    and not (mass[a][:, a]-mass[b][:, b]).nnz \
                    and not (stiffness[a][:, a]-stiffness[b][:, b]).nnz:
                self.parts = (a, b)
        diagonal = np.asarray(mass.diagonal())
        pressure_operator = (divergence @ sp.diags(1/diagonal) @ divergence.T).tocsc()
        if gauge:
            # Anchor only the auxiliary pressure preconditioner, not an original
            # equation. The outer system retains every B row and its mean gauge.
            free = np.ones(self.np);free[0] = 0.
            pressure_operator = (sp.diags(free) @ pressure_operator @ sp.diags(free)
                                 +sp.diags(1-free)).tocsc()
        started = perf_counter()
        self.pressure = _AMGInverse(PETSc, pressure_operator, kind, cycles=2)
        self.stats: dict[str, int | float] = dict(pressure_setups=1, velocity_setups=0,
            setup_wall_seconds=perf_counter()-started, preconditioner_calls=0,
            preconditioner_wall_seconds=0., shared_velocity_components=len(self.parts))

    def update(self, mass: float, viscosity: float) -> None:
        self.alpha, self.beta = mass+viscosity*self.reaction, viscosity
        key = (self.alpha, self.beta)
        reuse = self.anchor is not None and all(
            (x == y == 0.) or (x > 0 and y > 0 and .5 <= x/y <= 2.)
            for x, y in zip(key, self.anchor))
        if self.velocity is not None and reuse:
            return
        if self.velocity is not None:
            self.velocity.close()
        started = perf_counter()
        operator = mass*self.mass+viscosity*self.stiffness
        if self.parts:
            operator = operator[self.parts[0]][:, self.parts[0]]
        self.velocity = _AMGInverse(self.PETSc, operator,
            'jacobi' if viscosity == 0. else self.kind, cycles=2 if mass == 0. else 1)
        self.anchor = key
        self.stats['velocity_setups'] += 1
        self.stats['setup_wall_seconds'] += perf_counter()-started

    def apply(self, pc: Any, x: Any, y: Any) -> None:
        started = perf_counter()
        self.stats['preconditioner_calls'] += 1
        # The outer operator is row-scaled for coefficient-residual resolution.
        rhs = x.getArray(readonly=True)/self.scale
        rp = rhs[self.nv:self.nv+self.np].copy()
        multiplier = 0.
        if self.gauge:
            multiplier = float(np.sum(rp)/np.sum(self.weights))
            rp -= self.weights*multiplier
        pressure_rhs = rp.copy()
        if self.gauge:
            pressure_rhs[0] = 0.
        pressure = self.alpha*self.pressure.solve(pressure_rhs)+self.beta*rp/self.weights
        if self.gauge:
            pressure += (rhs[-1]-float(self.weights @ pressure))/float(np.sum(self.weights))
        velocity_rhs = rhs[:self.nv]+self.B.T @ pressure
        assert self.velocity is not None
        if self.parts:
            velocity = np.empty(self.nv)
            for part in self.parts:
                velocity[part] = self.velocity.solve(velocity_rhs[part])
        else:
            velocity = self.velocity.solve(velocity_rhs)
        output = y.getArray()
        output[:self.nv] = velocity
        output[self.nv:self.nv+self.np] = pressure
        if self.gauge:
            output[-1] = multiplier
        self.stats['preconditioner_wall_seconds'] += perf_counter()-started

    def close(self) -> None:
        if self.velocity is not None:
            self.velocity.close()
        self.pressure.close()


class PETScStokes:
    """Two/multiple RHS share block preconditioners, independently of SAV algebra."""
    def __init__(self, mass: Any, stiffness: Any, divergence: Any, weights: Array, *,
                 pressure_gauge: bool, reaction_coefficient: float = 0.,
                 components: Sequence[np.ndarray] = (), amg_type: str = 'gamg') -> None:
        PETSc = self.PETSc = _petsc()
        if amg_type not in ('gamg', 'hypre'):
            raise ValueError('AMG type must be gamg or hypre')
        if amg_type == 'hypre' and not PETSc.Sys.hasExternalPackage('hypre'):
            raise RuntimeError('This PETSc build has no Hypre support')
        self.block = _BlockPreconditioner(PETSc, mass, stiffness, divergence, weights,
            pressure_gauge, reaction_coefficient, components, amg_type)
        self.ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        self.ksp.setType('fgmres')
        self.ksp.setGMRESRestart(80)
        self.ksp.setTolerances(rtol=1e-12, atol=0., max_it=400)
        self.ksp.setErrorIfNotConverged(False)
        pc = self.ksp.getPC();pc.setType('python');pc.setPythonContext(self.block)
        size = mass.shape[0]+divergence.shape[0]+int(pressure_gauge)
        self.rhs = PETSc.Vec().createSeq(size, comm=PETSc.COMM_SELF)
        self.value = self.rhs.duplicate()
        self.matrix: Any = None
        self.csr: Any = None
        self.key: tuple[float, float] | None = None
        self.stats: dict[str, int | float] = dict(solves=0, iterations=0, failures=0,
                                                 solve_wall_seconds=0., matrix_updates=0)

    def solve(self, matrix: Any, rhs: Array, *, mass: float, viscosity: float) -> Array:
        key = mass, viscosity
        self.block.update(mass, viscosity)
        if key != self.key:
            previous = self.matrix
            scaled = sp.diags(self.block.scale) @ matrix
            self.matrix, self.csr = _matrix(self.PETSc, scaled)
            self.ksp.setOperators(self.matrix)
            if previous is not None:
                previous.destroy()
            self.key = key
            self.stats['matrix_updates'] += 1
        result = np.empty_like(rhs)
        started = perf_counter()
        for j in range(rhs.shape[1]):
            if not np.any(rhs[:, j]):
                result[:, j] = 0.
                continue
            self.rhs.getArray()[:] = self.block.scale*rhs[:, j]
            self.value.set(0.)
            self.ksp.solve(self.rhs, self.value)
            self.stats['solves'] += 1
            self.stats['iterations'] += self.ksp.getIterationNumber()
            if self.ksp.getConvergedReason() < 0:
                self.stats['failures'] += 1
                raise RuntimeError(f'PETSc FGMRES failed: {self.ksp.getConvergedReason()}')
            result[:, j] = self.value.getArray(readonly=True)
        self.stats['solve_wall_seconds'] += perf_counter()-started
        return result

    def metrics(self) -> dict[str, float | int]:
        return {**self.stats, **self.block.stats}

    def close(self) -> None:
        self.ksp.destroy();self.rhs.destroy();self.value.destroy()
        if self.matrix is not None:
            self.matrix.destroy()
        self.block.close()

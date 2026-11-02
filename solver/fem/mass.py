"""Exact consistent mass inverse, sharing identical uncoupled component factors."""
from collections.abc import Sequence
from typing import Any
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, splu

from .space import Array


class VelocityMassFactor:
    def __init__(self, matrix: Any, components: Sequence[np.ndarray], *, backend: str = 'direct') -> None:
        if backend not in ('direct', 'cg', 'metis'):
            raise ValueError('Unknown consistent mass backend')
        self.backend = backend
        self.components: tuple[np.ndarray, ...] = ()
        first = components[0]
        scalar = matrix[first][:, first].tocsc()
        if len(components) == 2 and len(first) == len(components[1]) \
                and len(first)*2 == matrix.shape[0] \
                and not matrix[first][:, components[1]].nnz \
                and not matrix[components[1]][:, first].nnz \
                and not (scalar-matrix[components[1]][:, components[1]]).nnz:
            self.components = tuple(components)
        else:
            scalar = matrix
        self.factor: Any = None
        self.ordering_wall_seconds = 0.
        self.ordering_cache_hit = 0
        if backend == 'metis':
            from .ordering import PermutedFactor, mass_nested_dissection
            started = perf_counter()
            permutation, hit = mass_nested_dissection(scalar)
            self.ordering_wall_seconds = perf_counter()-started
            self.ordering_cache_hit = int(hit)
            raw = splu(scalar[permutation][:, permutation].tocsc(), permc_spec='NATURAL', diag_pivot_thresh=0.)
            self.factor = PermutedFactor(raw, permutation)
            factor_nnz = raw.nnz
        elif backend == 'direct':
            self.factor = splu(scalar, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.)
            factor_nnz = self.factor.nnz
        else:
            factor_nnz = 0
        self.rows, self.nnz = scalar.shape[0], factor_nnz
        self.matrix_nonzeros = scalar.nnz
        self.matrix = scalar if backend == 'cg' else None
        self.preconditioner = None
        if backend == 'cg':
            diagonal = np.asarray(scalar.diagonal())
            if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.):
                raise ValueError('Consistent mass needs positive finite diagonal')
            self.preconditioner = LinearOperator(scalar.shape, matvec=lambda v: v/diagonal, dtype=float)
        self._calls, self._iterations, self._wall = 0, 0, 0.

    def metrics(self) -> dict[str, int | float]:
        return dict(solves=self._calls, cg_iterations=self._iterations, solve_wall_seconds=self._wall,
                    matrix_nonzeros=self.matrix_nonzeros, shared_components=len(self.components),
                    ordering_wall_seconds=self.ordering_wall_seconds, ordering_cache_hit=self.ordering_cache_hit)

    def _solve(self, rhs: Array) -> Array:
        if self.factor is not None:
            return self.factor.solve(rhs)
        assert self.matrix is not None
        columns = rhs[:, None] if rhs.ndim == 1 else rhs
        result = np.empty_like(columns)
        def iteration(_: Array) -> None:
            self._iterations += 1
        for j in range(columns.shape[1]):
            value, info = cg(self.matrix, columns[:, j], M=self.preconditioner,
                             rtol=1e-13, atol=0., maxiter=200, callback=iteration)
            if info != 0 or not np.isfinite(value).all():
                raise RuntimeError(f'Consistent mass CG did not converge: {info}')
            result[:, j] = value
        return result[:, 0] if rhs.ndim == 1 else result

    def solve(self, rhs: Array) -> Array:
        started = perf_counter()
        self._calls += 1
        try:
            if not self.components:
                return self._solve(rhs)
            columns = rhs[:, None] if rhs.ndim == 1 else rhs
            count = columns.shape[1]
            solved = self._solve(np.concatenate([columns[part] for part in self.components], axis=1))
            result = np.empty_like(columns)
            for component, part in enumerate(self.components):
                result[part] = solved[:, component*count:(component+1)*count]
            return result[:, 0] if rhs.ndim == 1 else result
        finally:
            self._wall += perf_counter()-started

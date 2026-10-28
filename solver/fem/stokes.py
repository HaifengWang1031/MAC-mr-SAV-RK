"""Exact shifted Stokes, integral gauge/full checks and optional recycled-factor GMRES."""
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres, splu
from .space import Array


@dataclass
class StokesSolution:
    velocity: Array
    pressure: Array
    residual: float
    continuity_residual: float


class FEMStokes:
    def __init__(self, mass_matrix: Any, stiffness: Any, divergence: Any, pressure_weights: Array,
                 *, tolerance: float = 1e-9, cache_size: int = 4, pressure_gauge: bool = True,
                 backend: str = 'direct', mass_solve: Callable[[Array], Array] | None = None) -> None:
        if type(pressure_gauge) is not bool or not np.isfinite(tolerance) or tolerance <= 0 \
                or type(cache_size) is not int or cache_size < 1 or backend not in ('direct', 'recycled'):
            raise ValueError('Invalid solver controls')
        self.mass, self.stiffness, self.divergence = mass_matrix, stiffness, divergence
        self.pressure_weights, self.pressure_gauge = pressure_weights, pressure_gauge
        self.tolerance, self.cache_size, self.backend = tolerance, cache_size, backend
        diagonal = np.asarray(self.mass.diagonal())
        if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.):
            raise ValueError('Consistent velocity mass must have positive finite diagonal')
        self._mass_solve = (splu(self.mass, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.).solve
                            if mass_solve is None else mass_solve)
        self._row_scale = np.r_[1/diagonal, np.ones(self.divergence.shape[0]+int(pressure_gauge))]
        self.cache: OrderedDict[tuple[float, float], Any] = OrderedDict()
        self._robust: set[tuple[float, float]] = set()
        self._matrix_key: tuple[float, float] | None = None
        self._current_matrix: Any = None
        self._stats: dict[str, float | int] = {
            'calls': 0, 'linear_columns': 0, 'factorizations': 0, 'cache_hits': 0,
            'recycled_solves': 0, 'recycle_refreshes': 0, 'krylov_iterations': 0,
            'preconditioner_solves': 0, 'triangular_solves': 0, 'fallbacks': 0,
            'krylov_norm_limit_hits': 0, 'factor_wall_seconds': 0., 'solve_wall_seconds': 0.,
            'max_coefficient_residual': 0., 'max_continuity_residual': 0.}

    def metrics(self) -> dict[str, float | int]:
        return {**self._stats, 'cached_factors': len(self.cache)}

    def _matrix(self, key: tuple[float, float]) -> Any:
        if key != self._matrix_key:
            mass, viscosity = key
            h = mass*self.mass+viscosity*self.stiffness
            if self.pressure_gauge:
                c = sp.csc_matrix(self.pressure_weights[:, None])
                # Retain every pressure-test row and the original integral constraint.
                self._current_matrix = sp.bmat([[h, -self.divergence.T, None],
                                                 [self.divergence, None, c],
                                                 [None, c.T, None]], format='csc')
            else:
                self._current_matrix = sp.bmat([[h, -self.divergence.T],
                                                 [self.divergence, None]], format='csc')
            self._matrix_key = key
        return self._current_matrix

    def _factor(self, key: tuple[float, float], matrix: Any, *, robust: bool = False) -> Any:
        factor, last_error = None, None
        choices = [('COLAMD', 1.)] if robust else [('MMD_AT_PLUS_A', 0.), ('COLAMD', 1.)]
        for ordering, pivot in choices:
            start = perf_counter()
            self._stats['factorizations'] += 1
            try:
                factor = splu(matrix, permc_spec=ordering, diag_pivot_thresh=pivot)
            except RuntimeError as error:
                last_error = error
            finally:
                self._stats['factor_wall_seconds'] += perf_counter()-start
            if factor is not None:
                break
            if ordering != 'COLAMD':
                self._stats['fallbacks'] += 1
        if factor is None:
            raise RuntimeError('Singular FEM Stokes system; check mesh resolution') from last_error
        self.cache[key] = factor
        self.cache.move_to_end(key)
        self._robust.discard(key)
        if ordering == 'COLAMD':
            self._robust.add(key)
        if len(self.cache) > self.cache_size:
            evicted, _ = self.cache.popitem(last=False)
            self._robust.discard(evicted)
        return factor

    def _solve_factor(self, factor: Any, rhs: Array, *, preconditioner: bool = False) -> Array:
        start = perf_counter()
        self._stats['triangular_solves'] += 1
        self._stats['preconditioner_solves'] += int(preconditioner)
        try:
            return factor.solve(rhs)
        finally:
            self._stats['solve_wall_seconds'] += perf_counter()-start

    def _checks(self, solved: Array, rhs: Array, load: Array, prescribed: Array,
                key: tuple[float, float]) -> tuple[Array, Array, Array, Array]:
        if not np.isfinite(solved).all():
            bad = np.full(rhs.shape[1], np.inf)
            return bad, bad, bad, bad
        nv, npres = self.mass.shape[0], self.divergence.shape[0]
        velocity, pressure = solved[:nv], solved[nv:nv+npres]
        mass, viscosity = key
        defect = mass*(self.mass @ velocity)+viscosity*(self.stiffness @ velocity) \
            - self.divergence.T @ pressure-load
        weak = np.max(np.abs(defect), axis=0)/(1+np.max(np.abs(load), axis=0))
        # This independent complete Bv check cannot be hidden by a gauge multiplier.
        continuity = np.max(np.abs(self.divergence @ velocity-prescribed), axis=0)
        gauge = np.abs(self.pressure_weights @ pressure) if self.pressure_gauge else np.zeros(rhs.shape[1])
        coefficient = np.max(np.abs(self._mass_solve(defect)), axis=0)/(1+np.max(np.abs(rhs), axis=0))
        return weak, continuity, gauge, coefficient

    @staticmethod
    def _valid(checks: tuple[Array, Array, Array, Array], tolerance: float) -> bool:
        return all(np.isfinite(value).all() and np.all(value <= tolerance) for value in checks)

    def _recycle(self, key: tuple[float, float], matrix: Any, system_rhs: Array) -> Array | None:
        # Exact current matrix; an old nearby factor is only a preconditioner, never a quantized shift.
        eligible = [old for old in self.cache if old[0] == key[0] and old[1] > 0. and key[1] > 0.
                    and abs(key[1]-old[1])/old[1] <= .05]
        if not eligible:
            return None
        anchor = min(eligible, key=lambda old: abs(key[1]-old[1])/old[1])
        factor = self.cache[anchor]
        self.cache.move_to_end(anchor)
        scale = self._row_scale
        operator = LinearOperator(matrix.shape, matvec=lambda v: scale*(matrix @ v), dtype=float)
        preconditioner = LinearOperator(matrix.shape,
            matvec=lambda v: self._solve_factor(factor, v/scale, preconditioner=True), dtype=float)
        result = np.zeros_like(system_rhs)

        def count_iteration(_: float) -> None:
            self._stats['krylov_iterations'] += 1

        try:
            for k in range(system_rhs.shape[1]):
                if not np.any(system_rhs[:, k]):
                    continue
                value, info = gmres(operator, scale*system_rhs[:, k], M=preconditioner,
                                    rtol=min(1e-11, self.tolerance*.01), atol=0., restart=10, maxiter=2,
                                    callback=count_iteration, callback_type='pr_norm')
                if info < 0:
                    return None
                if info > 0:
                    self._stats['krylov_norm_limit_hits'] += 1
                result[:, k] = value
        except (RuntimeError, ValueError, FloatingPointError):
            return None
        return result

    def solve_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float,
                      divergence_rhs: Array | None = None) -> list[StokesSolution]:
        if not np.isfinite([mass, viscosity]).all() or min(mass, viscosity) < 0 or mass+viscosity <= 0:
            raise ValueError('Require nonnegative finite shifts with nonzero total')
        if not len(columns):
            return []
        nv, npres = self.mass.shape[0], self.divergence.shape[0]
        rhs = np.column_stack(columns)
        if rhs.shape != (nv, len(columns)) or not np.isfinite(rhs).all():
            raise ValueError('Invalid velocity coefficients')
        prescribed = (np.zeros((npres, len(columns))) if divergence_rhs is None
                      else np.asarray(divergence_rhs, dtype=float))
        if prescribed.shape != (npres, len(columns)) or not np.isfinite(prescribed).all():
            raise ValueError('Invalid full continuity right-hand side')
        self._stats['calls'] += 1
        self._stats['linear_columns'] += len(columns)
        load = self.mass @ rhs
        key = float(mass), float(viscosity)
        matrix = self._matrix(key)
        system_rhs = np.vstack([load, prescribed])
        if self.pressure_gauge:
            system_rhs = np.vstack([system_rhs, np.zeros((1, len(columns)))])
        solved, checks = None, None
        if key in self.cache:
            self._stats['cache_hits'] += 1
            self.cache.move_to_end(key)
        elif self.backend == 'recycled':
            solved = self._recycle(key, matrix, system_rhs)
            if solved is not None:
                checks = self._checks(solved, rhs, load, prescribed, key)
                if self._valid(checks, min(self.tolerance*.1, 1e-10)):
                    self._stats['recycled_solves'] += 1
                else:
                    solved, checks = None, None
                    self._stats['recycle_refreshes'] += 1
        if solved is None:
            factor = self.cache[key] if key in self.cache else self._factor(key, matrix)
            try:
                solved = self._solve_factor(factor, system_rhs)
            except RuntimeError:
                if key in self._robust:
                    raise
                self._stats['fallbacks'] += 1
                solved = self._solve_factor(self._factor(key, matrix, robust=True), system_rhs)
            checks = self._checks(solved, rhs, load, prescribed, key)
        assert checks is not None
        if not self._valid(checks, self.tolerance) and key not in self._robust:
            self._stats['fallbacks'] += 1
            solved = self._solve_factor(self._factor(key, matrix, robust=True), system_rhs)
            checks = self._checks(solved, rhs, load, prescribed, key)
        if not self._valid(checks, self.tolerance):
            raise RuntimeError(f'FEM Stokes residual={np.max(checks[0]):.3e}, '
                               f'continuity={np.max(checks[1]):.3e}, pressure integral={np.max(checks[2]):.3e}, '
                               f'coefficient residual={np.max(checks[3]):.3e}')
        self._stats['max_coefficient_residual'] = max(self._stats['max_coefficient_residual'], float(np.max(checks[3])))
        self._stats['max_continuity_residual'] = max(self._stats['max_continuity_residual'], float(np.max(checks[1])))
        return [StokesSolution(solved[:nv, k].copy(), solved[nv:nv+npres, k].copy(),
                               float(checks[0][k]), float(checks[1][k])) for k in range(len(columns))]

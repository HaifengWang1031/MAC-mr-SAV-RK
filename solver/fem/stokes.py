"""Exact shifted Stokes with checked batched factor recycling and GMRES fallback."""
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres, splu
from .ordering import PermutedFactor, nested_dissection
from .space import Array

_Checks = tuple[Array, Array, Array, Array]


@dataclass
class StokesSolution:
    velocity: Array
    pressure: Array
    residual: float
    continuity_residual: float


class FEMStokes:
    def __init__(self, mass_matrix: Any, stiffness: Any, divergence: Any, pressure_weights: Array,
                 *, tolerance: float = 1e-9, cache_size: int = 4, pressure_gauge: bool = True,
                 backend: str = 'direct', mass_solve: Callable[[Array], Array] | None = None,
                 ordering: str = 'auto', factor_backend: str = 'superlu', threads: int | None = None,
                 reaction_coefficient: float = 0., components: Sequence[np.ndarray] = (),
                 amg_type: str = 'gamg') -> None:
        if type(pressure_gauge) is not bool or not np.isfinite(tolerance) or tolerance <= 0 \
                or type(cache_size) is not int or cache_size < 1 or backend not in ('direct', 'recycled', 'petsc') \
                or not np.isfinite(reaction_coefficient) or reaction_coefficient < 0. \
                or amg_type not in ('gamg', 'hypre') \
                or ordering not in ('auto', 'mmd', 'metis') or factor_backend not in ('superlu', 'pardiso') \
                or (factor_backend == 'pardiso' and ordering != 'auto') \
                or (threads is not None and (type(threads) is not int or threads < 1 or factor_backend != 'pardiso')):
            raise ValueError('Invalid solver controls')
        self.mass, self.stiffness, self.divergence = mass_matrix, stiffness, divergence
        self.pressure_weights, self.pressure_gauge = pressure_weights, pressure_gauge
        self.tolerance, self.cache_size, self.backend = tolerance, cache_size, backend
        self.ordering = ordering
        self.factor_backend, self.threads = factor_backend, threads
        if factor_backend == 'pardiso':
            from .pardiso import require_pardiso
            require_pardiso()
        self._permutation: np.ndarray | None = None
        self._ordering_attempted = False
        diagonal = np.asarray(self.mass.diagonal())
        if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.):
            raise ValueError('Consistent velocity mass must have positive finite diagonal')
        self._mass_solve = (splu(self.mass, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.).solve
                            if mass_solve is None else mass_solve)
        self._row_scale = np.r_[1/diagonal, np.ones(self.divergence.shape[0]+int(pressure_gauge))]
        self._mass_row_bound = float(np.max(np.asarray(abs(self.mass).sum(axis=1)).ravel()/diagonal))
        self.cache: OrderedDict[tuple[float, float], Any] = OrderedDict()
        self._robust: set[tuple[float, float]] = set()
        self._matrix_key: tuple[float, float] | None = None
        self._current_matrix: Any = None
        self._stats: dict[str, float | int] = {
            'calls': 0, 'linear_columns': 0, 'factorizations': 0, 'cache_hits': 0,
            'recycled_solves': 0, 'recycle_refreshes': 0, 'krylov_iterations': 0,
            'preconditioner_solves': 0, 'triangular_solves': 0, 'fallbacks': 0,
            'krylov_norm_limit_hits': 0, 'factor_wall_seconds': 0., 'solve_wall_seconds': 0.,
            'defect_solves': 0, 'defect_iterations': 0, 'defect_failures': 0, 'gmres_solves': 0,
            'ordering_preparations': 0, 'ordering_cache_hits': 0, 'ordering_failures': 0,
            'ordering_wall_seconds': 0., 'metis_factorizations': 0, 'factor_nnz_peak': 0,
            'pardiso_factorizations': 0,
            'max_coefficient_residual': 0., 'max_continuity_residual': 0.,
            'petsc_attempts': 0, 'petsc_checked_solves': 0, 'petsc_fallbacks': 0,
            'petsc_refinements': 0}
        self._multigrid: Any = None
        if backend == 'petsc':
            from .multigrid import PETScStokes
            self._multigrid = PETScStokes(self.mass, self.stiffness, self.divergence,
                self.pressure_weights, pressure_gauge=pressure_gauge,
                reaction_coefficient=reaction_coefficient, components=components, amg_type=amg_type)

    def metrics(self) -> dict[str, float | int]:
        extra = {} if self._multigrid is None else {
            'petsc_'+key: value for key, value in self._multigrid.metrics().items()}
        return {**self._stats, **extra, 'cached_factors': len(self.cache)}

    def close(self) -> None:
        """Release an optional external solver; ordinary LU ownership is unchanged."""
        if self._multigrid is not None:
            self._multigrid.close()
            self._multigrid = None

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
        factor: Any = None
        last_error = None
        use_metis = self.ordering == 'metis' or (self.ordering == 'auto'
                    and self.factor_backend == 'superlu' and matrix.shape[0] >= 20000)
        if use_metis and not robust and not self._ordering_attempted:
            self._ordering_attempted = True
            self._stats['ordering_preparations'] += 1
            start = perf_counter()
            try:
                self._permutation, hit = nested_dissection(self.mass, self.stiffness, self.divergence,
                    self.pressure_weights if self.pressure_gauge else None)
                self._stats['ordering_cache_hits'] += int(hit)
            except (ImportError, OSError, RuntimeError, ValueError):
                self._stats['ordering_failures'] += 1
                self._stats['fallbacks'] += 1
            finally:
                self._stats['ordering_wall_seconds'] += perf_counter()-start
        choices = [('COLAMD', 1.)] if robust else [('MMD_AT_PLUS_A', 0.), ('COLAMD', 1.)]
        if not robust and self._permutation is not None:
            choices.insert(0, ('NATURAL', 0.))
        if not robust and self.factor_backend == 'pardiso':
            choices.insert(0, ('PARDISO', 0.))
        for ordering, pivot in choices:
            start = perf_counter()
            self._stats['factorizations'] += 1
            try:
                if ordering == 'PARDISO':
                    from .pardiso import PardisoFactor
                    raw = PardisoFactor(matrix, self.threads)
                    self._stats['pardiso_factorizations'] += 1
                else:
                    ordered = matrix if ordering != 'NATURAL' else matrix[self._permutation][:, self._permutation].tocsc()
                    raw = splu(ordered, permc_spec=ordering, diag_pivot_thresh=pivot)
                self._stats['factor_nnz_peak'] = max(self._stats['factor_nnz_peak'], int(getattr(raw, 'nnz', 0)))
                if ordering == 'NATURAL':
                    assert self._permutation is not None
                    factor = PermutedFactor(raw, self._permutation)
                    self._stats['metis_factorizations'] += 1
                else:
                    factor = raw
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

    def _checks(self, solved: Array, rhs: Array | None, load: Array, prescribed: Array,
                key: tuple[float, float]) -> _Checks:
        if not np.isfinite(solved).all():
            bad = np.full(load.shape[1], np.inf)
            return bad, bad, bad, bad
        nv, npres = self.mass.shape[0], self.divergence.shape[0]
        velocity, pressure = solved[:nv], solved[nv:nv+npres]
        mass, viscosity = key
        defect = mass*(self.mass @ velocity)+viscosity*(self.stiffness @ velocity) \
            - self.divergence.T @ pressure-load
        weak = np.max(np.abs(defect), axis=0)/(1+np.max(np.abs(load), axis=0))
        # This independent complete Bv check cannot be hidden by a gauge multiplier.
        continuity = np.max(np.abs(self.divergence @ velocity-prescribed), axis=0)
        gauge = np.abs(self.pressure_weights @ pressure) if self.pressure_gauge else np.zeros(load.shape[1])
        rhs_norm = self.load_norm_lower_bound(load) if rhs is None else np.max(np.abs(rhs), axis=0)
        coefficient = np.max(np.abs(self._mass_solve(defect)), axis=0)/(1+rhs_norm)
        return weak, continuity, gauge, coefficient

    @staticmethod
    def _valid(checks: _Checks, tolerance: float) -> bool:
        return all(np.isfinite(value).all() and np.all(value <= tolerance) for value in checks)

    def _anchor(self, key: tuple[float, float]) -> Any | None:
        eligible = [old for old in self.cache if old[0] == key[0] and old[1] > 0. and key[1] > 0.
                    and abs(key[1]-old[1])/old[1] <= .05]
        if not eligible:
            return None
        anchor = min(eligible, key=lambda old: abs(key[1]-old[1])/old[1])
        factor = self.cache[anchor]
        self.cache.move_to_end(anchor)
        return factor

    def _refine(self, key: tuple[float, float], matrix: Any, system_rhs: Array,
                rhs: Array | None, load: Array, prescribed: Array) -> tuple[Array | None, _Checks | None]:
        """Correct all RHS together with an old LU, accepting only the exact current equations.

        The cheap diagonal-scaled residual only screens when to run the complete
        checks. Since |diag(M)^-1 M e|_inf <= row_bound*|e|_inf, it cannot replace
        the consistent-mass coefficient check. Stagnation or failure falls back
        to the original Krylov path, then a fresh checked factorization.
        """
        factor = self._anchor(key)
        if factor is None:
            return None, None
        result = np.zeros_like(system_rhs)
        residual = system_rhs.copy()
        nv = self.mass.shape[0]
        denominator = 1.+(self.load_norm_lower_bound(load) if rhs is None else np.max(np.abs(rhs), axis=0))
        limit = min(self.tolerance*.1, 1e-10)
        previous = np.inf
        try:
            for _ in range(8):
                result += self._solve_factor(factor, residual, preconditioner=True)
                self._stats['defect_iterations'] += 1
                if not np.isfinite(result).all():
                    break
                residual = system_rhs-matrix @ result
                cheap = np.max(np.abs(self._row_scale[:nv, None]*residual[:nv]), axis=0)/denominator
                norm = float(np.max(cheap))
                if norm <= self._mass_row_bound*limit:
                    checks = self._checks(result, rhs, load, prescribed, key)
                    if self._valid(checks, limit):
                        self._stats['defect_solves'] += 1
                        return result, checks
                if norm >= .9*previous:
                    break
                previous = norm
        except (RuntimeError, ValueError, FloatingPointError):
            pass
        self._stats['defect_failures'] += 1
        return None, None

    def _recycle(self, key: tuple[float, float], matrix: Any, system_rhs: Array) -> Array | None:
        # Exact current matrix; an old nearby factor is only a preconditioner, never a quantized shift.
        factor = self._anchor(key)
        if factor is None:
            return None
        self._stats['gmres_solves'] += 1
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

    def load_norm_lower_bound(self, load: Array) -> Any:
        """Lower bound for ||M^-1 F||_inf, columnwise for batched loads."""
        scale = self._row_scale[:self.mass.shape[0]]
        return np.max(np.abs(scale*load if load.ndim == 1 else scale[:, None]*load), axis=0)/self._mass_row_bound

    def solve_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float,
                      divergence_rhs: Array | None = None) -> list[StokesSolution]:
        return self._solve_columns(columns, mass=mass, viscosity=viscosity,
                                   divergence_rhs=divergence_rhs, weak_load=False)

    def solve_load_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float,
                           divergence_rhs: Array | None = None) -> list[StokesSolution]:
        """Solve weak loads directly, retaining true coefficient/continuity checks."""
        return self._solve_columns(columns, mass=mass, viscosity=viscosity,
                                   divergence_rhs=divergence_rhs, weak_load=True)

    def _solve_columns(self, columns: Sequence[Array], *, mass: float, viscosity: float,
                      divergence_rhs: Array | None, weak_load: bool) -> list[StokesSolution]:
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
        load = rhs if weak_load else self.mass @ rhs
        checked_rhs = None if weak_load else rhs
        key = float(mass), float(viscosity)
        matrix = self._matrix(key)
        system_rhs = np.vstack([load, prescribed])
        if self.pressure_gauge:
            system_rhs = np.vstack([system_rhs, np.zeros((1, len(columns)))])
        solved, checks = None, None
        if self.backend == 'petsc':
            self._stats['petsc_attempts'] += 1
            try:
                solved = self._multigrid.solve(matrix, system_rhs, mass=mass, viscosity=viscosity)
                checks = self._checks(solved, checked_rhs, load, prescribed, key)
                # A Krylov norm is not the original coefficient criterion. Correct
                # the exact mixed defect before resorting to a global LU; each
                # correction retains every continuity row and the same gauge.
                for _ in range(2):
                    if self._valid(checks, self.tolerance):
                        break
                    self._stats['petsc_refinements'] += 1
                    defect = system_rhs-matrix @ solved
                    solved += self._multigrid.solve(matrix, defect, mass=mass, viscosity=viscosity)
                    checks = self._checks(solved, checked_rhs, load, prescribed, key)
                if self._valid(checks, self.tolerance):
                    self._stats['petsc_checked_solves'] += 1
                else:
                    solved, checks = None, None
            except RuntimeError:
                solved, checks = None, None
            if solved is None:
                self._stats['petsc_fallbacks'] += 1
                self._stats['fallbacks'] += 1
        elif key in self.cache:
            self._stats['cache_hits'] += 1
            self.cache.move_to_end(key)
        elif self.backend == 'recycled':
            solved, checks = self._refine(key, matrix, system_rhs, checked_rhs, load, prescribed)
            if solved is None:
                solved = self._recycle(key, matrix, system_rhs)
            if solved is not None:
                if checks is None:
                    checks = self._checks(solved, checked_rhs, load, prescribed, key)
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
            checks = self._checks(solved, checked_rhs, load, prescribed, key)
        assert checks is not None
        if not self._valid(checks, self.tolerance) and key not in self._robust:
            self._stats['fallbacks'] += 1
            solved = self._solve_factor(self._factor(key, matrix, robust=True), system_rhs)
            checks = self._checks(solved, checked_rhs, load, prescribed, key)
        if not self._valid(checks, self.tolerance):
            raise RuntimeError(f'FEM Stokes residual={np.max(checks[0]):.3e}, '
                               f'continuity={np.max(checks[1]):.3e}, pressure integral={np.max(checks[2]):.3e}, '
                               f'coefficient residual={np.max(checks[3]):.3e}')
        self._stats['max_coefficient_residual'] = max(self._stats['max_coefficient_residual'], float(np.max(checks[3])))
        self._stats['max_continuity_residual'] = max(self._stats['max_continuity_residual'], float(np.max(checks[1])))
        return [StokesSolution(solved[:nv, k].copy(), solved[nv:nv+npres, k].copy(),
                               float(checks[0][k]), float(checks[1][k])) for k in range(len(columns))]

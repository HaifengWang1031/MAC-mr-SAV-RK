"""Cached library P2 shapes, batched component contractions and owned single-field cache."""
from time import perf_counter
from typing import Any

import numpy as np
from .space import Array


class P2Quadrature:
    def __init__(self, basis: Any) -> None:
        scalar = basis.split_bases()[0]
        self._parts = tuple(basis.split_indices())
        if len(self._parts) != 2 or any(not np.array_equal(basis.doflocs[:, part], scalar.doflocs)
                                        for part in self._parts):
            raise ValueError('Unsupported vector/scalar finite-element degree-of-freedom mapping')
        self._dofs = scalar.element_dofs
        self._phi = np.stack([np.asarray(field[0]) for field in scalar.basis])
        self._grad = np.stack([np.asarray(field[0].grad) for field in scalar.basis])
        self._weights, self._ndof, self._scalar_ndof = basis.dx, basis.N, scalar.N
        self._last: Array | None = None
        self._field: tuple[Array, Array] | None = None
        self._convection: Array | None = None
        self._stats: dict[str, float | int] = {
            'field_evaluations': 0, 'field_cache_hits': 0, 'convection_assemblies': 0,
            'convection_cache_hits': 0, 'load_assemblies': 0,
            'field_wall_seconds': 0., 'convection_wall_seconds': 0., 'load_wall_seconds': 0.}

    def metrics(self) -> dict[str, float | int]:
        return self._stats.copy()

    def evaluate(self, full: Array) -> tuple[Array, Array]:
        start = perf_counter()
        try:
            if self._last is not None and np.array_equal(full, self._last):
                self._stats['field_cache_hits'] += 1
                assert self._field is not None
                return self._field
            coefficients = np.stack([full[part][self._dofs] for part in self._parts])
            velocity = np.einsum('ike,keq->ieq', coefficients, self._phi)
            gradient = np.einsum('ike,kjeq->ijeq', coefficients, self._grad)
            velocity.flags.writeable = gradient.flags.writeable = False
            self._last, self._field, self._convection = full.copy(), (velocity, gradient), None
            self._stats['field_evaluations'] += 1
            return velocity, gradient
        finally:
            self._stats['field_wall_seconds'] += perf_counter()-start

    def _scatter(self, local: Array) -> Array:
        result = np.zeros(self._ndof)
        for component, part in enumerate(self._parts):
            result[part] = np.bincount(self._dofs.ravel(), weights=local[component].ravel(),
                                       minlength=self._scalar_ndof)
        return result

    def load(self, values: Array) -> Array:
        start = perf_counter()
        try:
            if values.shape != (2, *self._weights.shape):
                raise ValueError('Invalid vector quadrature samples')
            self._stats['load_assemblies'] += 1
            local = np.einsum('ieq,keq->ike', values*self._weights[None], self._phi)
            return self._scatter(local)
        finally:
            self._stats['load_wall_seconds'] += perf_counter()-start

    def convection_load(self, full: Array) -> Array:
        start = perf_counter()
        try:
            velocity, gradient = self.evaluate(full)
            if self._convection is not None:
                self._stats['convection_cache_hits'] += 1
                return self._convection.copy()
            advective = np.einsum('ijeq,jeq->ieq', gradient, velocity)
            test_advective = np.einsum('kjeq,jeq->keq', self._grad, velocity)
            local = .5*(np.einsum('ieq,keq->ike', advective*self._weights[None], self._phi)
                        -np.einsum('ieq,keq->ike', velocity*self._weights[None], test_advective))
            result = self._scatter(local)
            # Open-flow callers add a facet load; never lend them cached mutable storage.
            self._convection = result.copy()
            self._stats['convection_assemblies'] += 1
            return result
        finally:
            self._stats['convection_wall_seconds'] += perf_counter()-start

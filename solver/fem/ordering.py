"""Shift-independent structural ordering and coordinate restoration for sparse LU."""
from collections import OrderedDict
import hashlib
from threading import Lock
from typing import Any

import numpy as np
import scipy.sparse as sp

from .space import Array

_CACHE: OrderedDict[bytes, np.ndarray] = OrderedDict()
_LOCK = Lock()


def _pattern(matrix: Any) -> Any:
    result = matrix.copy()
    result.eliminate_zeros()
    result.data = np.ones(result.nnz)
    return result


def nested_dissection(mass: Any, stiffness: Any, divergence: Any,
                      pressure_weights: Array | None) -> tuple[np.ndarray, bool]:
    """Order the union of all shifted patterns, sharing at most four permutations.

    The off-diagonal saddle blocks have opposite signs: a numerical A+A.T
    would cancel them. Build an unsigned structural union instead, including
    both mass and stiffness even when the first requested shift has one zero.
    """
    h = _pattern(mass)+_pattern(stiffness)
    b = _pattern(divergence)
    if pressure_weights is None:
        graph = sp.bmat([[h, b.T], [b, None]], format='csr')
    else:
        c = _pattern(sp.csc_matrix(pressure_weights[:, None]))
        graph = sp.bmat([[h, b.T, None], [b, None, c], [None, c.T, None]], format='csr')
    graph.data[:] = 1.
    graph = (graph+graph.T).tocsr()
    graph.setdiag(0.)
    graph.eliminate_zeros()
    graph.sort_indices()
    return _order_graph(graph)


def mass_nested_dissection(matrix: Any) -> tuple[np.ndarray, bool]:
    """Order the scalar consistent mass graph without forming a saddle system."""
    graph = _pattern(matrix).tocsr()
    graph = (graph+graph.T).tocsr()
    graph.setdiag(0.)
    graph.eliminate_zeros()
    graph.sort_indices()
    return _order_graph(graph)


def _order_graph(graph: Any) -> tuple[np.ndarray, bool]:
    digest = hashlib.sha256()
    digest.update(np.asarray(graph.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(graph.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(graph.indices, dtype=np.int64).tobytes())
    key = digest.digest()
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key], True
        import pymetis
        dtype = pymetis.zero_copy_dtype()
        adjacency = pymetis.CSRAdjacency(graph.indptr.astype(dtype), graph.indices.astype(dtype))
        permutation = np.asarray(pymetis.nested_dissection(
            adjacency=adjacency, options=pymetis.Options(seed=0))[0], dtype=np.intp)
        if not np.array_equal(np.sort(permutation), np.arange(graph.shape[0])):
            raise ValueError('Invalid nested-dissection permutation')
        permutation.flags.writeable = False
        _CACHE[key] = permutation
        if len(_CACHE) > 4:
            _CACHE.popitem(last=False)
        return permutation, False


class PermutedFactor:
    """Keep all public vectors in the original finite-element coordinates."""
    def __init__(self, factor: Any, permutation: np.ndarray) -> None:
        self.factor, self.permutation = factor, permutation

    def solve(self, rhs: Array) -> Array:
        ordered = self.factor.solve(rhs[self.permutation])
        result = np.empty_like(ordered)
        result[self.permutation] = ordered
        return result

"""Optional oneMKL general sparse factor, with local thread limits and checked RHS."""
from collections.abc import Iterator
from contextlib import contextmanager
import ctypes
from threading import RLock
from typing import Any

from .space import Array


def require_pardiso() -> None:
    try:
        import pypardiso  # noqa: F401 -- optional native dependency is loaded only on request.
    except (ImportError, OSError) as error:
        raise ValueError('PARDISO requires the pardiso extra on Linux x86_64 (uv sync --extra pardiso)') from error


class PardisoFactor:
    """Own an immutable CSR system; restore the caller's MKL thread setting.

    The velocity-pressure saddle matrix is general nonsymmetric (mtype=11).
    Two internal refinement steps are retained: fewer failed the original
    physical-pressure checks in the qualified cylinder experiment.
    """
    def __init__(self, matrix: Any, threads: int | None = None) -> None:
        from pypardiso import PyPardisoSolver
        self.matrix = matrix.tocsr(copy=True)
        self.matrix.sort_indices()
        self.solver = PyPardisoSolver(mtype=11)
        self._lock, self.threads = RLock(), threads
        # Use the C export. The lowercase compatibility symbol in oneMKL
        # expects a Fortran address and crashes when passed an integer value.
        self._set_threads = self.solver.libmkl.MKL_Set_Num_Threads_Local
        self._set_threads.argtypes, self._set_threads.restype = [ctypes.c_int], ctypes.c_int
        with self._thread_limit():
            try:
                self.solver.factorize(self.matrix)
            except Exception as error:
                raise RuntimeError('PARDISO factorization failed') from error
        self.solver.set_iparm(25, 2)
        self.solver.set_iparm(8, 2)
        self.nnz = max(0, int(self.solver.get_iparm(18)))

    @contextmanager
    def _thread_limit(self) -> Iterator[None]:
        previous = self._set_threads(self.threads) if self.threads is not None else None
        try:
            yield
        finally:
            if previous is not None:
                self._set_threads(previous)

    def solve(self, rhs: Array) -> Array:
        with self._lock, self._thread_limit():
            try:
                return self.solver.solve(self.matrix, rhs)
            except Exception as error:
                raise RuntimeError('PARDISO solve failed') from error

    def __del__(self) -> None:
        try:
            self.solver.free_memory(everything=True)
        except Exception:
            pass  # Best-effort release also covers a partially constructed native factor.

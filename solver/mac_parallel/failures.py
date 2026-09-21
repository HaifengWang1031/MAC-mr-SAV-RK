"""Agreement boundaries for local-only work; never wrap MPI/PETSc collectives."""
from typing import Any, Callable, TypeVar

T = TypeVar('T')

class ParallelFailure(RuntimeError):
    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failures = failures
        super().__init__('; '.join(f"rank {f['rank']} [{f['phase']}]: {f['error']}" for f in failures))


def local_call(comm: Any, phase: str, action: Callable[[], T]) -> T:
    """Run only local work, then agree before anybody enters the next collective.

    Every rank must call this boundary in the same order. An action must contain no
    collectives, including logically collective PETSc vector allocation/destruction.
    Process loss and failures inside MPI itself are outside this recovery contract.
    """
    failure = None
    value: Any = None
    try:
        value = action()
    except Exception as exc:
        failure = {'rank': comm.rank, 'phase': phase, 'error': f'{type(exc).__name__}: {exc}'}
    failures = [item for item in comm.allgather(failure) if item is not None]
    if failures:
        raise ParallelFailure(failures)
    return value

"""Resume identity: separate modules that can change accepted numbers from readers.

A chain may be legally resumed only when the *numerical* source is byte-identical:
the solver, the record writer, and the config/problem/stepping modules. Read-only
readers and analysers (`chain.py`, `analyze.py`, `stats.py`, `inventory.py`,
`plan.py`) must not be able to block continuation, because they cannot change a
single accepted value. The full aggregate provenance hash is still recorded per
segment, so a resumed chain reports every harness version that touched it.
"""
import hashlib
from pathlib import Path

from experiments.records import PROJECT, digest
from run_support.source import CORE_SOURCE_DIRS

# Only files that determine the accepted trajectory, the sampling of it, or the
# on-disk representation of those numbers.
NUMERICAL_MODULES = (
    'experiments/records.py',
    'experiments/inverse_cascade/config.py',
    'experiments/inverse_cascade/problem.py',
    'experiments/inverse_cascade/segments.py',
)


def numerical_files(root: Path = PROJECT) -> list[Path]:
    solver = [path for name in CORE_SOURCE_DIRS for path in sorted((root/name).rglob('*.py'))]
    explicit = [root/rel for rel in NUMERICAL_MODULES]
    missing = [str(path) for path in explicit if not path.is_file()]
    if missing or any(not list((root/name).rglob('*.py')) for name in CORE_SOURCE_DIRS):
        raise ValueError(f'Numerical source inventory is incomplete: {missing}')
    return solver+explicit


def numerical_digest(root: Path = PROJECT) -> str:
    """Stable digest of the numerical source inventory, independent of harness code."""
    lines = [f'{path.relative_to(root).as_posix()}\0{digest(path)}\n'
             for path in numerical_files(root)]
    return hashlib.sha256(''.join(sorted(lines)).encode()).hexdigest()

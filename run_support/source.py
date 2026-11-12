"""Explicit core/task source inventories and the transitional legacy policy.

Core provenance never discovers local research implicitly. Callers must name the
files or Python directories that determine their task. Paths are relative to the
source root; missing, empty or escaping inventories fail closed.
"""
from collections.abc import Sequence
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
ENVIRONMENT_FILES = ('pyproject.toml', 'uv.lock', '.python-version')
CORE_SOURCE_DIRS = ('solver', 'run_support')
# Order is part of the historical byte-stream encoding. Append, do not reorder.
LEGACY_SOURCE_DIRS = ('solver', 'experiments', 'tools', 'run_support', 'postprocess')


def digest(path: Path) -> str:
    """Hash a file without loading its full contents into memory."""
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _source_path(root: Path, path: Path) -> Path:
    try:
        resolved = path.resolve()
    except RuntimeError as error:
        raise ValueError(f'Invalid source symlink: {path}') from error
    if not resolved.is_relative_to(root):
        raise ValueError(f'Source path escapes root: {path}')
    # Source copies must preserve importable names. Resolving a symlink and
    # copying only its target would silently remove that name from the copy.
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError(f'Source symlinks are unsupported: {ancestor}')
    return resolved


def source_files(*, root: Path = PROJECT,
                 task_paths: Sequence[str | Path] = ()) -> tuple[Path, ...]:
    """Return the deterministic, complete core plus explicitly named task files.

    Directories contribute Python files only; explicit files may also capture
    task data/configuration. The recorded inventory uses portable relative names.
    """
    root = root.resolve()
    files: set[Path] = set()
    for name in (*ENVIRONMENT_FILES, *CORE_SOURCE_DIRS, *task_paths):
        path = _source_path(root, root/name)
        if name in ENVIRONMENT_FILES and not path.is_file():
            raise ValueError(f'Missing source file: {name}')
        if name in CORE_SOURCE_DIRS and not path.is_dir():
            raise ValueError(f'Missing source directory: {name}')
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = []
            for entry in sorted(path.rglob('*')):
                if entry.is_symlink():
                    # rglob does not descend into directory symlinks: inspect
                    # them explicitly rather than silently dropping code.
                    _source_path(root, entry)
                if entry.suffix == '.py':
                    candidates.append(entry)
            if not candidates:
                raise ValueError(f'Empty source directory: {name}')
        else:
            raise ValueError(f'Missing source path: {name}')
        for candidate in candidates:
            resolved = _source_path(root, candidate)
            if not resolved.is_file():
                raise ValueError(f'Missing source file: {candidate}')
            files.add(resolved)
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def legacy_source_files(root: Path = PROJECT) -> tuple[Path, ...]:
    """Ordered broad inventory for old callers and source-selected workers.

    Missing directories are allowed, as in the old implementation, so genuine
    older trees without run_support/postprocess can still be captured. Required
    environment files remain mandatory at hashing/copy time.
    """
    files = [root/name for name in ENVIRONMENT_FILES]
    for directory in LEGACY_SOURCE_DIRS:
        files.extend(sorted((root/directory).rglob('*.py')))
    return tuple(files)


def _environment() -> dict[str, Any]:
    return {'python': sys.version, 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in
                         ('numpy', 'scipy', 'numba', 'h5py', 'sympy', 'matplotlib')},
            'threads': {key: os.environ.get(key) for key in (
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')}}


def provenance(*, root: Path = PROJECT,
               task_paths: Sequence[str | Path] = ()) -> dict[str, Any]:
    """Identify core plus explicit task source using versioned, auditable files.

    This is a new identity policy, not a relabelling of historical provenance.
    The source root's absolute location does not enter the code fingerprint.
    """
    root = root.resolve()
    inventory = {path.relative_to(root).as_posix(): digest(path)
                 for path in source_files(root=root, task_paths=task_paths)}
    hasher = hashlib.sha256(b'explicit-source-v2\0')
    for name, checksum in inventory.items():
        hasher.update(f'{name}\0{checksum}\n'.encode())
    environment = _environment()
    environment['packages'].update({name: importlib.metadata.version(name)
                                    for name in ('scikit-fem', 'pymetis')})
    return {'source_schema': 2, 'source_files': inventory,
            'code_sha256': hasher.hexdigest(), **environment}


def legacy_provenance(*, root: Path = PROJECT) -> dict[str, Any]:
    """Keep legacy encoding while including the newly extracted implementation.

    No saved source/identity is rewritten. New broad-scope runs change identity
    honestly when shared implementation changes, including run_support/postprocess.
    """
    hasher = hashlib.sha256()
    for path in legacy_source_files(root):
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(path.read_bytes())
    return {'code_sha256': hasher.hexdigest(), **_environment()}

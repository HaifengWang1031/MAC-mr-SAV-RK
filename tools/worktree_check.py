"""Read-only worktree identity preflight; never executes subsequent writes."""
from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys


RESEARCH_PATHS = ('experiments', 'tests/tasks', 'tools/benchmark_steps.py',
                  'tools/benchmark_fem_adaptive.py', 'tools/probe_hele_shaw_inlet.py',
                  'tools/rerun_low_high_stability_probe.py', 'tools/search_low_high_stability_steps.py')


@dataclass(frozen=True)
class Worktree:
    """Observed local identity, not authorization or remote synchronization."""
    cwd: str
    root: str
    branch: str | None
    head: str
    upstream: str | None
    python: str
    python_prefix: str
    profile: str
    research_paths: tuple[str, ...]
    dirty: bool


def _git(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(['git', '-c', 'core.fsmonitor=false', *arguments], cwd=cwd,
                            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'},
                            text=True, capture_output=True, timeout=15, check=False)
    if result.returncode:
        raise ValueError(result.stderr.strip() or 'Git inspection failed')
    return result.stdout.strip()


def inspect_worktree(cwd: Path | None = None) -> Worktree:
    """Read Git identity and research-layout presence, not scientific qualification."""
    for variable in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR'):
        if variable in os.environ:
            raise ValueError('Cannot verify current worktree with ambient '+variable)
    where = (Path.cwd() if cwd is None else cwd).resolve(strict=True)
    root = Path(_git(where, 'rev-parse', '--show-toplevel')).resolve(strict=True)
    branch = _git(where, 'branch', '--show-current') or None
    head = _git(where, 'rev-parse', 'HEAD')
    upstream = (_git(where, 'for-each-ref', '--format=%(upstream:short)',
                     'refs/heads/'+branch) or None) if branch else None
    present = tuple(name for name in RESEARCH_PATHS
                    if (root/name).exists() or (root/name).is_symlink())
    local_sources = root/'experiments'
    local_tests = root/'tests/tasks'
    full_local = (local_sources.is_dir() and not local_sources.is_symlink() and
                  any(local_sources.rglob('*.py')) and local_tests.is_dir() and
                  not local_tests.is_symlink() and any(local_tests.rglob('test_*.py')))
    profile = 'core' if not present else ('full-local' if full_local else 'mixed')
    return Worktree(str(where), str(root), branch, head, upstream,
                    os.path.abspath(sys.executable), sys.prefix, profile, present,
                    bool(_git(where, 'status', '--porcelain=v1', '--untracked-files=normal')))


def main(argv: Sequence[str] | None = None) -> int:
    """Print JSON and exit nonzero unless actual identity matches explicit expectations."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--expect-root', required=True, type=Path)
    parser.add_argument('--expect-branch', required=True)
    parser.add_argument('--expect-head', required=True)
    parser.add_argument('--profile', required=True, choices=('core', 'full-local'))
    parser.add_argument('--expect-python', type=Path)
    parser.add_argument('--expect-upstream')
    args = parser.parse_args(argv)
    try:
        worktree = inspect_worktree()
        errors: list[str] = []
        if not args.expect_root.expanduser().is_absolute():
            errors.append('Expected root must be an absolute path')
        if not Path(worktree.cwd).is_relative_to(Path(worktree.root)):
            errors.append('Current directory is outside the reported Git root')
        expected_head = args.expect_head.lower()
        if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', expected_head):
            errors.append('Expected HEAD must be a full commit object id')
        expected = {'root': str(args.expect_root.expanduser().resolve()),
                    'branch': args.expect_branch, 'head': expected_head, 'profile': args.profile}
        if args.expect_python is not None:
            # Resolving executable symlinks would conflate different virtualenvs.
            expected['python'] = os.path.abspath(args.expect_python.expanduser())
        if args.expect_upstream is not None:
            expected['upstream'] = args.expect_upstream
        for field, value in expected.items():
            actual = getattr(worktree, field)
            if actual != value:
                errors.append(f'{field} mismatch: expected {value!r}, observed {actual!r}')
        if worktree.dirty:
            errors.append('Worktree is dirty; inspect pending changes before starting a new write workflow')
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({'ok': False, 'worktree': None, 'errors': [str(exc)]}, ensure_ascii=False))
        return 2
    print(json.dumps({'ok': not errors, 'worktree': asdict(worktree), 'errors': errors},
                     ensure_ascii=False))
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Run the core suite in an isolated copy with no local research packages.

Usage: python -m tests.core_distribution [--collect-only]
The same interpreter/environment is reused; no dependencies are installed.
"""
from argparse import ArgumentParser
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[1]

_RUN = '''
import importlib.abc
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
assert not (root/'experiments').exists() and not (root/'tools').exists()
assert not (root/'tests/tasks').exists()

class NoResearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'experiments', 'tools'} or fullname.startswith('tests.tasks'):
            raise AssertionError('Core test imported research: '+fullname)

sys.meta_path.insert(0, NoResearch())
import pytest
raise SystemExit(pytest.main(['tests/core', '-q', '-p', 'no:cacheprovider', *sys.argv[2:]]))
'''


def check_core(*, collect_only: bool = False) -> subprocess.CompletedProcess[str]:
    """Execute actual tests (or collection) without falling back to this checkout."""
    with TemporaryDirectory(prefix='mac-core-tests-') as temporary:
        root = Path(temporary)
        ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbc', '*.nbi')
        for name in ('solver', 'run_support', 'postprocess', 'tests/core', 'tests/references'):
            shutil.copytree(PROJECT/name, root/name, ignore=ignore)
        for name in ('pyproject.toml', 'uv.lock', '.python-version',
                     'tests/__init__.py', 'tests/core_distribution.py'):
            shutil.copy2(PROJECT/name, root/name)
        # -I removes checkout/PYTHONPATH leakage. Disable unrelated installed
        # pytest plugins; the core suite only needs pytest's built-in fixtures.
        environment = {**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
        arguments = ['--collect-only'] if collect_only else []
        return subprocess.run([sys.executable, '-I', '-c', _RUN, str(root), *arguments],
                              cwd=root, env=environment, text=True, capture_output=True,
                              timeout=300, check=False)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--collect-only', action='store_true')
    result = check_core(collect_only=parser.parse_args().collect_only)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

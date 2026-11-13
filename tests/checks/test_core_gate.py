"""The real isolated CLI must enforce the index, not only parse it in unit tests."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('collect_only', [True, False], ids=['collection', 'execution'])
def test_isolated_suite_rejects_a_required_regression_moved_to_tasks(tmp_path, collect_only):
    source = tmp_path/'source'
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbc', '*.nbi')
    for name in ('solver', 'run_support', 'postprocess', 'tests/core', 'tests/references'):
        shutil.copytree(PROJECT/name, source/name, ignore=ignore)
    for name in ('pyproject.toml', 'uv.lock', '.python-version', 'tests/__init__.py',
                 'tests/core_distribution.py', 'tests/core_requirements.py',
                 'tests/core_requirements.json'):
        shutil.copy2(PROJECT/name, source/name)
    moved = source/'tests/tasks/test_adaptive_continuation.py'
    moved.parent.mkdir()
    (source/'tests/core/test_adaptive_continuation.py').rename(moved)
    arguments = ['--collect-only'] if collect_only else []
    result = subprocess.run([sys.executable, '-m', 'tests.core_distribution', *arguments],
                            cwd=source, env={**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'},
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 4, result.stdout[-1500:]+'\n'+result.stderr
    assert 'Required core regressions not collected' in result.stderr
    assert 'test_bounded_history_and_joint_pi_continuation' in result.stderr

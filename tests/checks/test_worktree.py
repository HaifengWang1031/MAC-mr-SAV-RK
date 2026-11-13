"""Read-only CLI checks against real temporary Git repositories."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


CLI = Path(__file__).resolve().parents[2]/'tools/worktree_check.py'


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path/'repo'
    root.mkdir()
    git(root, 'init', '--quiet', '--initial-branch=main')
    (root/'README.md').write_text('Known fixture\n')
    git(root, 'add', 'README.md')
    git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'fixture')
    return root


def check_command(repo, *extra):
    return [sys.executable, '-B', str(CLI), '--expect-root', str(repo),
            '--expect-branch', 'main', '--expect-head', git(repo, 'rev-parse', 'HEAD'),
            '--profile', 'core', '--expect-python', sys.executable, *extra]


def run_check(repo, *extra, cwd=None, env=None):
    return subprocess.run(check_command(repo, *extra), cwd=cwd or repo, env=env, text=True,
                          capture_output=True, timeout=30)


def fingerprint(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(),
                                      p.stat().st_mode, p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_matching_worktree_reports_identity_without_writing_files(repo):
    before = fingerprint(repo)
    result = run_check(repo)
    assert result.returncode == 0, result.stderr+result.stdout
    report = json.loads(result.stdout)
    assert report['ok'] is True and report['errors'] == []
    observed = report['worktree']
    assert observed['root'] == observed['cwd'] == str(repo.resolve())
    assert observed['branch'] == 'main' and observed['head'] == git(repo, 'rev-parse', 'HEAD')
    assert observed['upstream'] is None and observed['profile'] == 'core'
    assert observed['python'] == os.path.abspath(sys.executable)
    assert observed['python_prefix'] == sys.prefix
    assert observed['dirty'] is False and observed['research_paths'] == []
    assert fingerprint(repo) == before


@pytest.mark.parametrize('layout,profile,code', [
    ('full', 'full-local', 0), ('partial', 'full-local', 2),
    ('empty', 'full-local', 2), ('tool-only', 'core', 2),
])
def test_profiles_report_presence_and_reject_partial_research_layout(repo, layout, profile, code):
    (repo/'.gitignore').write_text('experiments/\ntests/tasks/\ntools/benchmark_steps.py\n')
    git(repo, 'add', '.gitignore')
    git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'local ignores')
    if layout == 'tool-only':
        (repo/'tools').mkdir()
        (repo/'tools/benchmark_steps.py').write_text('# local computation\n')
    else:
        (repo/'experiments').mkdir()
        if layout != 'empty':
            (repo/'experiments/__init__.py').write_text('')
        if layout != 'partial':
            (repo/'tests/tasks').mkdir(parents=True)
            if layout == 'full':
                (repo/'tests/tasks/test_local.py').write_text('def test_local(): pass\n')
    result = run_check(repo, '--profile', profile)
    assert result.returncode == code, result.stdout+result.stderr
    observed = json.loads(result.stdout)['worktree']
    assert observed['profile'] == ('full-local' if layout == 'full' else 'mixed')
    assert observed['dirty'] is False


def test_upstream_can_be_bound_without_contacting_a_remote(repo):
    git(repo, 'remote', 'add', 'origin', str(repo.parent/'not-a-network-remote'))
    git(repo, 'update-ref', 'refs/remotes/origin/main', git(repo, 'rev-parse', 'HEAD'))
    git(repo, 'branch', '--set-upstream-to=origin/main')
    good = run_check(repo, '--expect-upstream', 'origin/main')
    assert good.returncode == 0, good.stdout+good.stderr
    assert json.loads(good.stdout)['worktree']['upstream'] == 'origin/main'
    bad = run_check(repo, '--expect-upstream', 'origin/research')
    assert bad.returncode == 2 and 'upstream mismatch' in bad.stdout


def test_non_repository_fails_as_json_instead_of_falling_back_to_another_tree(repo):
    result = run_check(repo, cwd=repo.parent)
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report['ok'] is False and report['worktree'] is None and report['errors']


def test_ambient_git_repository_override_is_rejected(repo):
    result = run_check(repo, env={**os.environ, 'GIT_DIR': str(repo/'.git')})
    assert result.returncode == 2, result.stdout+result.stderr
    assert 'GIT_DIR' in result.stdout


def test_failed_preflight_blocks_a_success_chained_write(repo):
    marker = repo.parent/'write must not happen'
    command = shlex.join(check_command(repo, '--expect-branch', 'research'))
    result = subprocess.run(['sh', '-c', command+' && touch '+shlex.quote(str(marker))],
                            cwd=repo, text=True, capture_output=True, timeout=30)
    assert result.returncode == 2, result.stdout+result.stderr
    assert not marker.exists()


@pytest.mark.parametrize('flag,value,error', [
    ('--expect-root', '/', 'root'),
    ('--expect-root', '.', 'absolute'),
    ('--expect-branch', 'research', 'branch'),
    ('--expect-head', '0'*40, 'head'),
    ('--expect-head', 'HEAD', 'full commit'),
    ('--profile', 'full-local', 'profile'),
])
def test_mismatched_expectations_fail_without_mutating_the_worktree(repo, flag, value, error):
    before = fingerprint(repo)
    result = run_check(repo, flag, value)
    assert result.returncode == 2, result.stdout+result.stderr
    report = json.loads(result.stdout)
    assert report['ok'] is False
    assert any(error in item.lower() for item in report['errors'])
    assert fingerprint(repo) == before


def test_interpreter_check_keeps_distinct_virtualenv_paths_distinct(repo):
    other_python = repo.parent/'other-venv/bin/python'
    other_python.parent.mkdir(parents=True)
    other_python.symlink_to(Path(sys.executable).resolve())
    result = run_check(repo, '--expect-python', str(other_python))
    assert result.returncode == 2, result.stdout+result.stderr
    assert any('python' in error.lower() for error in json.loads(result.stdout)['errors'])


@pytest.mark.parametrize('state', ['dirty', 'detached'])
def test_dirty_or_detached_worktrees_are_not_ready_for_writes(repo, state):
    if state == 'dirty':
        (repo/'README.md').write_text('Uncommitted change\n')
    else:
        git(repo, 'checkout', '--detach', '--quiet')
    before = fingerprint(repo)
    result = run_check(repo)
    assert result.returncode == 2, result.stdout+result.stderr
    assert fingerprint(repo) == before

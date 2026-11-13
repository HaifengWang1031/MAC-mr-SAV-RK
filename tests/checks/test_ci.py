"""CI's artifact contract; JSON syntax is also valid YAML, without a YAML dependency."""
import json
from pathlib import Path
import re


PROJECT = Path(__file__).resolve().parents[2]


def test_ci_runs_locked_public_checks_without_suppressing_failures():
    workflow = json.loads((PROJECT/'.github/workflows/core.yml').read_text())
    assert set(workflow['on']) == {'pull_request', 'push', 'workflow_dispatch'}
    assert workflow['on']['push'] == {'branches': ['main']}
    assert workflow['permissions'] == {'contents': 'read'}
    job = workflow['jobs']['core']
    assert not job.get('if') and not job.get('continue-on-error')
    steps = job['steps']
    commands = [step for step in steps if 'run' in step]
    assert [step['run'] for step in commands] == [
        'uv sync --locked --group dev',
        'uv run --locked mypy',
        'uv run --locked pytest -q -ra',
        'uv run --locked python -m tests.core_distribution',
    ]
    assert all(not step.get('if') and not step.get('continue-on-error') for step in commands)
    actions = [step for step in steps if 'uses' in step]
    assert len(actions) == 2
    assert all(re.fullmatch(r'(actions/checkout|astral-sh/setup-uv)@[0-9a-f]{40}',
                            step['uses']) for step in actions)
    assert actions[0]['uses'].startswith('actions/checkout@')
    assert actions[0]['with']['persist-credentials'] is False
    assert actions[1]['uses'].startswith('astral-sh/setup-uv@')
    assert actions[1]['with']['python-version'] == '3.12'

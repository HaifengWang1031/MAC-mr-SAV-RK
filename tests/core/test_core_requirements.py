"""Critical mechanism availability, not total test counts or coverage inference."""
from contextlib import nullcontext
import json

import pytest

from tests.core_requirements import check_core_requirements


@pytest.mark.parametrize('collected,missing', [
    (['tests/core/test_example.py::test_resume[bounded]',
      'tests/core/test_other.py::test_unregistered'], False),
    ([], True),
    (['tests/tasks/test_example.py::test_resume'], True),
])
def test_registered_core_regression_must_be_collected_in_core(tmp_path, collected, missing):
    manifest = tmp_path/'required.json'
    manifest.write_text(json.dumps({'continuation': ['tests/core/test_example.py::test_resume']}))
    with pytest.raises(ValueError, match='not collected.*test_resume') if missing else nullcontext():
        check_core_requirements(collected, manifest=manifest)


@pytest.mark.parametrize('requirements', [
    {},
    {'resume': ['tests/tasks/test_example.py::test_resume']},
    {'resume': ['tests/core/test_example.py::test_resume',
                'tests/core/test_example.py::test_resume']},
])
def test_invalid_or_duplicate_core_requirements_fail_closed(tmp_path, requirements):
    manifest = tmp_path/'required.json'
    manifest.write_text(json.dumps(requirements))
    with pytest.raises(ValueError):
        check_core_requirements([], manifest=manifest)

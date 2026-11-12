"""Distribution seam: core tests collect with research absent and imports blocked."""
from tests.core_distribution import check_core


def test_core_suite_collects_without_research_packages() -> None:
    result = check_core(collect_only=True)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert 'tests collected' in result.stdout

"""Paired capture/run/load analysis verifies provenance without re-running solvers."""
import json

import pytest

from experiments.legendre_optimization.analyze import analyze
from experiments.legendre_optimization.run import CONFIG, capture_source, run_benchmark


def test_paired_benchmark_persists_sources_and_analyzes_only_saved_fields(tmp_path, monkeypatch):
    baseline = tmp_path/'baseline'
    capture_source(baseline)
    config = {**json.loads(CONFIG.read_text()), 'legendre_modes': [4], 'schemes': ['imex_euler_mrsav'],
              'steps': 2, 'repeats': 1, 'micro_repeats': 1, 'measure_memory': False,
              'error_quadrature': 24, 'error_quadrature_refined': 48}
    directory = run_benchmark(config, baseline, root=tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('analysis started a solver')
    monkeypatch.setattr('experiments.legendre_optimization.run.build_model', forbidden)
    report = analyze(directory, root=tmp_path)
    result = json.loads((report/'analysis.json').read_text())
    assert result['all_equivalence_passed'] and len(result['pairs']) == 2
    assert (directory/'baseline_source/model.py').exists()
    assert (directory/'optimized_source/model.py').exists()
    (directory/'fields.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        analyze(directory, root=tmp_path)

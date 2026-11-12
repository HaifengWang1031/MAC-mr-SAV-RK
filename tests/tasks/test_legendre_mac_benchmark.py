"""Independent reconstruction check and saved-run integrity/analysis seam."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.legendre_mac_benchmark.analyze import analyze
from experiments.legendre_mac_benchmark.run import CONFIG, mac_evaluate, run_benchmark, schedules
from solver.mac.grid import MACGrid


def test_mac_reconstruction_matches_bilinear_interior_and_zero_walls():
    grid = MACGrid(8, 6, 1.3, .8)
    xu, yu = grid.coordinates("u")
    xv, yv = grid.coordinates("v")
    u, v = (1+xu)*(2+yu), (3+xv)*(4+yv)
    u[:, [0, -1]], v[[0, -1], :] = 0., 0.
    velocity = grid.pack(u, v)
    x, y = np.array([.4, .7, .9]), np.array([.2, .35, .5])
    reconstructed = mac_evaluate(grid, velocity, x, y)
    xx, yy = np.meshgrid(x, y)
    np.testing.assert_allclose(reconstructed[0], (1+xx)*(2+yy), rtol=1e-14)
    np.testing.assert_allclose(reconstructed[1], (3+xx)*(4+yy), rtol=1e-14)
    for component in mac_evaluate(grid, velocity, np.array([0., grid.lx]), np.array([0., grid.ly])):
        assert np.all(component == 0.)


def test_benchmark_save_load_analyze_and_reject_corruption(tmp_path, monkeypatch):
    config = {**json.loads(CONFIG.read_text()), "mac_grids": [6], "legendre_modes": [4],
              "schemes": ["sdirk3"], "steps": 2, "repeats": 1, "micro_repeats": 1,
              "error_quadrature": 32, "error_quadrature_refined": 64}
    steps = schedules(config)
    assert steps["fixed"].sum() == pytest.approx(steps["variable"].sum())
    directory = run_benchmark(config, root=tmp_path)
    manifest = json.loads((directory/"manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["completed_cases"] == 6
    # The analyzer must operate on saved fields, never invoke the computation path.
    def forbidden(*args, **kwargs):
        raise AssertionError("analysis launched computation")
    monkeypatch.setattr("experiments.legendre_mac_benchmark.run.run_case", forbidden)
    report = analyze(directory, root=tmp_path)
    analysis = json.loads((report/"analysis.json").read_text())
    assert len(analysis["summaries"]) == 6
    assert analysis["mac_tensor_vs_direct_relative_difference_max"] < 1e-9
    assert (report/"time_accuracy.png").exists()
    assert (report/"timings.csv").exists()
    (directory/"results.json").write_text("[]")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        analyze(directory, root=tmp_path)

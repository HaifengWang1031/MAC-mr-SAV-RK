"""Periodic MHD research entry point, reuse, saved corrections and readonly plots."""
import h5py
import numpy as np

from experiments.periodic_mhd.analyze import analyze
from experiments.periodic_mhd.records import load_mhd
from experiments.periodic_mhd.run import run, run_study


def test_orszag_tang_run_reuse_and_readonly_comparison(tmp_path):
    config = {'physical': {'log_every': 1000}, 'time_grid': 12, 'curve_grid': 18,
              'time_steps': [1/16, 1/32], 'reference_dt': 1/128, 'curve_dt': 1/32}
    batch = run_study(config, root=tmp_path)
    settings = {'log_every': 1000, 'nx': 18, 'ny': 18, 'dt': 1/32, 'scheme': 'sdirk3_mrsav'}
    directory = run(settings, root=tmp_path)
    assert str(directory) in batch.read_text()
    cfg, manifest = load_mhd(directory, require_complete=True)
    assert manifest['metrics']['accepted_steps'] == 64
    with h5py.File(directory/'results.h5', 'r') as data:
        assert data['snapshots/magnetic'].shape == (3, 2, cfg['ny'], cfg['nx'])
        assert np.max(data['diagnostics/magnetic_divergence_inf'][:]) < 1e-12
        assert np.max(data['stages/scalar_residual'][:]) < 1e-12
        values = data['corrections/values'][:]
        np.testing.assert_allclose(data['corrections/frozen_factor'][:],
                                   1+values[..., 0]+values[..., 0]**2)
        np.testing.assert_array_equal(data['stages/r'][:], values[..., 1])
        assert np.max(np.abs(values)) > 1e-5
    report = tmp_path/'reports'
    summary = analyze(batch, report)
    assert '2' in summary['time_convergence']
    assert summary['paper_curves']['1']['dissipation']['sdirk3']['peak'] > .07
    assert (report/'time_convergence.png').is_file()
    assert (report/'paper_comparison.png').is_file()

"""Independent spectral benchmark and public MAC TS-mode growth."""
import h5py
import numpy as np

from experiments.channel_ns.animate import read_movie, periodic_shift
from experiments.channel_ns.instability import growth_record
from experiments.channel_ns.problems import build_model
from experiments.channel_ns.run import effective_config, run
from experiments.channel_ns.ts_mode import continuous_eigenvalue, discrete_mode


def test_continuous_orr_sommerfeld_matches_published_benchmark():
    known = .0037396706229799-.2375264888204682j
    assert abs(continuous_eigenvalue(10000., 1., 80)-known) < 2e-8
    assert abs(continuous_eigenvalue(30000., .8, 80)-continuous_eigenvalue(30000., .8, 100)) < 2e-6


def test_mac_seed_normalisation_constraints_saved_growth_and_time_refinement(tmp_path):
    cfg = {'case': 'ts_wave', 'backend': 'fft', 'scheme': 'sdirk3', 'nx': 32, 'ny': 96,
           'lx': 2*np.pi/.8, 'ly': 2., 'nu': 1/30000., 'driving': 2/30000.,
           'epsilon': 1e-5, 'T': 2., 'dt': .05, 'snapshots': [0., 1., 2.]}
    model, initial, _ = build_model(effective_config(cfg))
    mode = discrete_mode(model.grid, cfg['nu'], cfg['driving'])
    assert mode.eigenvalue.real > 0 and mode.eigen_residual < 1e-8
    assert np.max(np.abs(model.apply_D(model.vector(initial)))) < 1e-12
    u, v = model.collocated_velocity(mode.velocity)
    np.testing.assert_allclose(np.max(np.hypot(u, v)), 1., atol=1e-13)
    rows = []
    for step in (.05, .025):
        directory = run({**cfg, 'dt': step}, root=tmp_path)
        data = read_movie(directory)
        assert data.initial_mode is not None
        with h5py.File(directory/'results.h5') as stored:
            assert stored['initial_mode'].attrs['reynolds_half_height'] == 30000.
        row = growth_record(directory)
        assert row['amplitude_gain'] > 1.
        assert abs(row['growth_fit']-mode.eigenvalue.real) < 2e-7
        rows.append(row)
    assert abs(rows[0]['growth_fit']-rows[1]['growth_fit']) < 1e-7


def test_wave_coordinate_shift_has_known_sign_and_preserves_constant_field():
    x = 2*np.pi*np.arange(32)/32
    field = np.tile(np.cos(x), (4, 1))
    np.testing.assert_allclose(periodic_shift(field, 3.5), np.tile(np.cos(x+2*np.pi*3.5/32), (4, 1)), atol=1e-14)
    np.testing.assert_allclose(periodic_shift(np.ones_like(field), 3.5), 1.)

import h5py
import numpy as np

from experiments.periodic_mhd.analyze_zhou import analyze_zhou, snapshot_metrics
from experiments.periodic_mhd.records import load_mhd
from experiments.periodic_mhd.run import run
from solver.fourier import FourierSpace


def test_zhou_pair_preserves_initial_geometry_and_supports_readonly_analysis(tmp_path):
    base = {'case': 'zhou_2019', 'S0': 125, 'nx': 48, 'ny': 48,
            'T': .5, 'dt': 1/32, 'snapshots': [0., .25, .5]}
    ordinary = run(base, root=tmp_path)
    sav = run({**base, 'scheme': 'sdirk3_mrsav'}, root=tmp_path)
    cfg, _ = load_mhd(ordinary, require_complete=True)
    assert cfg['nu'] == cfg['eta'] == .001
    space = FourierSpace(48, 48)
    with h5py.File(ordinary/'results.h5') as h:
        np.testing.assert_array_equal(h['snapshots/actual_times'][:], base['snapshots'])
        flux = h['snapshots/flux'][0]
        b = h['snapshots/magnetic'][0]
        background = np.cos(8*space.x)*np.cos(8*space.y)/8
        bx = -np.cos(8*space.x)*np.sin(8*space.y)
        by = np.sin(8*space.x)*np.cos(8*space.y)
        np.testing.assert_allclose(np.linalg.norm(b-np.stack((bx, by)))/np.linalg.norm((bx, by)),
                                   cfg['noise_rms'], rtol=1e-9)
        np.testing.assert_allclose(h['snapshots/current'][0], 128*background, atol=.002)
        count, peak, spectrum = snapshot_metrics(space, flux, b)
        assert (count, peak) == (256, 11)
        np.testing.assert_allclose(spectrum.sum(), h['diagnostics/magnetic'][0], rtol=1e-12)

    output = tmp_path/'analysis'
    summary = analyze_zhou(ordinary, sav, output)
    assert summary['shared_initial_fields_verified']
    assert (output/'snapshot_diagnostics.csv').is_file()
    assert (output/'zhou_snapshots.png').is_file()

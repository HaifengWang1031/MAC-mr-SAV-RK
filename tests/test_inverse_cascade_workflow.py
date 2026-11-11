import json

import h5py
import numpy as np
import pytest

from experiments.inverse_cascade.analyze import analyze, block_statistics
from experiments.inverse_cascade.config import effective_config
from experiments.inverse_cascade.problem import paper_force, initial_state
from experiments.inverse_cascade.records import load
from experiments.inverse_cascade.run import run, run_scan
from solver.fourier import FourierSpace
from solver.fourier.forced_mhd import ForcedFourierMHD, flux_potential


def test_paper_sources_parameters_and_seed():
    s = FourierSpace(36, 36)
    kf, f0, mu = 3, 2., .25
    f = paper_force(s, kf, f0, mu)
    omega = s.to_physical(1j*(s.kx*f[0, 1]-s.ky*f[0, 0]))
    ga = s.to_physical(flux_potential(s, f[1]))
    np.testing.assert_allclose(omega, 2*f0*kf*np.cos(kf*s.x)*np.cos(kf*s.y), atol=1e-13)
    np.testing.assert_allclose(ga, mu*f0/kf*np.sin(kf*s.x)*np.sin(kf*s.y), atol=1e-14)
    np.testing.assert_allclose(np.sqrt(s.inner(f[1], f[1])/s.inner(f[0], f[0])), mu)
    cfg = effective_config({'nx': 36, 'ny': 36, 'kf': kf, 'f0': f0})
    np.testing.assert_allclose(cfg['nu_plus'], np.sqrt(f0)*kf**(-3.5)/cfg['Re_plus'])
    np.testing.assert_allclose(cfg['nu_minus'], np.sqrt(f0)*kf**4.5/cfg['Re_minus'])
    model = ForcedFourierMHD(s, cfg['nu_plus'], cfg['nu_minus'], force=lambda t: f)
    initial = initial_state(model, kf, f0, mu, 2014, .01)
    z = model.vector(initial)
    np.testing.assert_allclose([np.sqrt(s.inner(v, v)/s.area) for v in z], .01*np.sqrt(f0/kf)*np.array([1., mu]))
    assert model.physical_divergence_inf(z) < 1e-14
    assert np.max(np.abs(z[..., 0, 0])) < 1e-17
    # Same continuous trigonometric seed, not grid-sample white noise.
    fine = FourierSpace(72, 72)
    other = ForcedFourierMHD(fine, cfg['nu_plus'], cfg['nu_minus'], force=lambda t: paper_force(fine, kf, f0, mu))
    fine_state = initial_state(other, kf, f0, mu, 2014, .01)
    np.testing.assert_allclose(fine_state.u[::2, ::2], initial.u, atol=1e-16)
    for invalid in ({'mu': -1}, {'seed': True}, {'T': .015}, {'extra': 1}, {'kf': 100}):
        with pytest.raises(ValueError):
            effective_config(invalid)


def test_block_uncertainty_known_means():
    times = np.arange(21, dtype=float)
    values = np.r_[0., np.repeat([1., 2., 3., 4.], 5)]
    stats = block_statistics(times, values, start=0, duration=5)
    assert stats['block_means'] == [1., 2., 3., 4.]
    assert stats['mean'] == 2.5
    np.testing.assert_allclose(stats['prefix'][-1]['ci95_halfwidth'], 3.182446305*np.std([1., 2., 3., 4.], ddof=1)/2)
    assert stats['prefix'][2]['ci95_halfwidth'] is None
    with pytest.raises(ValueError):
        block_statistics(times, values, start=20, duration=5)


def test_run_load_reuse_scan_and_readonly_analysis(tmp_path, monkeypatch):
    cfg = {'nx': 24, 'ny': 24, 'kf': 2, 'T': .4, 'dt': .01, 'sample_every': 2}
    batch_path = run_scan({'base': cfg, 'members': [{'mu': 0}, {'mu': .4}]}, root=tmp_path)
    batch = json.loads(batch_path.read_text())
    paths = [__import__('pathlib').Path(v['run']) for v in batch['members']]
    assert batch['status'] == 'complete'
    assert run({**cfg, 'mu': .4}, root=tmp_path) == paths[1]
    c, m = load(paths[1], require_complete=True)
    assert m['metrics']['accepted_steps'] == 40
    with h5py.File(paths[1]/'results.h5', 'r') as d:
        assert d['spectra/Pi_E'].shape[0] == 21
        assert d['snapshots/magnetic'].shape == (2, 2, 24, 24)
        assert np.max(np.abs(d['spectra/Pi_E'][:, -1])) < 1e-13
        assert np.max(np.abs(d['spectra/Pi_Q'][:, -1])) < 1e-13
    def forbidden(*args, **kwargs):
        raise AssertionError('Read-only analysis must never integrate')
    monkeypatch.setattr('experiments.inverse_cascade.run.integrate', forbidden)
    summary = analyze(paths, tmp_path/'reports', burn_in=.1, block=.06, K=2)
    assert summary['runs'][0]['budgets']['Q']['mean_injection'] == 0
    assert summary['runs'][1]['statistics']['E']['std'] > 0
    assert (tmp_path/'reports'/'window_statistics.png').is_file()
    # A vector copy is emitted for the LaTeX manuscript; it must be a real PDF.
    vector = tmp_path/'reports'/'window_statistics.pdf'
    assert vector.is_file() and vector.read_bytes()[:5] == b'%PDF-'
    assert (tmp_path/'reports'/'statistics.csv').is_file()
    with (paths[1]/'config.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='checksum'):
        load(paths[1])


def test_coupling_gate_is_opt_in_and_records_failure(tmp_path):
    base = {'nx': 24, 'ny': 24, 'kf': 2, 'T': .4, 'dt': .01, 'sample_every': 2, 'mu': .4}
    plain = run(base, root=tmp_path)
    cfg, manifest = load(plain, require_complete=True)
    assert 'coupling_tolerance' not in cfg
    assert 'max_abs_coupling_minus_one' not in manifest['metrics']
    assert 'coupling_tolerance' not in effective_config(base)
    with pytest.raises(ValueError, match='SAV scheme'):
        effective_config({**base, 'scheme': 'sdirk3', 'coupling_tolerance': 1e-6})
    with pytest.raises(ValueError, match='positive'):
        effective_config({**base, 'coupling_tolerance': 0.})
    loose = run({**base, 'coupling_tolerance': 1.}, root=tmp_path)
    assert load(loose, require_complete=True)[1]['metrics']['max_abs_coupling_minus_one'] >= 0
    with pytest.raises(RuntimeError, match='coupling deviation'):
        run({**base, 'coupling_tolerance': 1e-15}, root=tmp_path)
    failed = [p for p in (tmp_path/'runs'/'inverse_cascade').iterdir()
              if json.loads((p/'manifest.json').read_text())['status'] == 'failed']
    assert len(failed) == 1
    assert 'coupling' in json.loads((failed[0]/'manifest.json').read_text())['error']
    assert not (failed[0]/'results.h5').exists()

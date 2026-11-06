import h5py
import numpy as np
import pytest

from experiments.periodic_mhd.analyze import analyze
from experiments.periodic_mhd.records import load_mhd
from experiments.periodic_mhd.run import orszag_tang_initial, run, run_study
from solver.fourier import FourierMHD, FourierSpace
from solver.integrate import integrate
from solver.schemes.registry import make_scheme


def test_projected_mhd_flux_and_joint_energy_pairing():
    s = FourierSpace(20, 19)
    model = FourierMHD(s, .02, .03)
    initial = orszag_tang_initial(model)
    nonlinear = s.to_physical(model.nonlinear(model.vector(initial)))
    # Differentiate the exact low modes in (7) and remove the gradient analytically.
    x, y = s.x, s.y
    expected = np.array([
        [-.6*np.sin(2*x)*np.cos(y), 1.2*np.cos(2*x)*np.sin(y)],
        [(np.sin(2*x)-np.sin(x))*np.cos(y), (np.cos(x)-2*np.cos(2*x))*np.sin(y)],
    ])
    np.testing.assert_allclose(nonlinear, expected, atol=5e-14)
    np.testing.assert_allclose(
        [model.diagnostics(initial)[key] for key in ('kinetic', 'magnetic', 'dissipation')],
        [.5, .5, .02+2.5*.03], atol=1e-14)
    random = np.random.default_rng(223).normal(size=(2, 2, *s.shape))
    z = s.project(s.to_spectral(random))
    n = model.nonlinear(z)
    assert model.physical_divergence_inf(n) < 1e-12
    assert abs(model.inner(n, z)) < 1e-12
    with pytest.raises(RuntimeError, match='divergence'):
        model.solve(s.to_spectral(random), mass=1., viscosity=.01)


@pytest.mark.parametrize('name', ('sdirk3', 'sdirk3_mrsav'))
def test_exact_independently_diffusing_periodic_shears(name):
    s = FourierSpace(17, 16)
    model = FourierMHD(s, .2, .07)
    u, bx = np.sin(s.y), np.cos(2*s.y)
    zero = np.zeros(s.shape)
    initial = model.state(0., s.to_spectral(np.array([[u, zero], [bx, zero]])))
    result = integrate(model, make_scheme(name), initial, [.01]*16)
    assert result.status == 'complete', result.error
    t = result.final.t
    np.testing.assert_allclose(result.final.u, np.exp(-.2*t)*u, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.final.magnetic[0], np.exp(-4*.07*t)*bx, rtol=0, atol=1e-10)
    assert abs(result.final.r) < 1e-14
    assert all(stage.pressure.size == 0 for stage in result.final_stages)


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

"""Same phase forcing on MAC: known limits, discrete curl identity and run records."""
import numpy as np
import pytest

from experiments.torque_transition.analyze_mac import audit_mac
from experiments.torque_transition.mac_model import make_mac_model, mac_observables
from experiments.torque_transition.preflight import run_probe


def test_mac_force_matches_x0_and_x12_and_continuum_torque():
    zero = make_mac_model(64, 0.)
    maximum = make_mac_model(128, 180.)
    assert mac_observables(zero, zero.state(0., np.zeros(zero.grid.size)), 0)['torque'] == pytest.approx(0., abs=1e-16)
    state = maximum.state(0., np.zeros(maximum.grid.size))
    assert mac_observables(maximum, state, 12)['torque'] == pytest.approx(.04, rel=.006)
    f0 = zero.grid.unpack(zero.force(0.))
    x, _ = zero.grid.coordinates('v')
    np.testing.assert_allclose(f0[1][1:-1, :], .1*np.sin(5*x[1:-1, :]), atol=1e-16)


def test_mac_vertex_curl_and_dissipation_obey_discrete_identity():
    model = make_mac_model(32, 45.)
    g = model.grid
    x, y = g.coordinates('vertex')
    psi = np.sin(x/2)**2*np.sin(y/2)**2
    psi[[0, -1], :] = 0.
    psi[:, [0, -1]] = 0.
    u, v = np.diff(psi, axis=0)/g.hy, -np.diff(psi, axis=1)/g.hx
    velocity = g.pack(u, v)
    row = mac_observables(model, model.state(0., velocity), 4)
    assert row['relative_divergence'] < 1e-14
    assert row['dissipation_minus_2nuZ'] == pytest.approx(0., abs=1e-16)


def test_mac_probe_keeps_representation_and_rejects_spectral_seed(tmp_path):
    directory = tmp_path/'mac'
    row = run_probe(directory, nodes=16, case=4, duration=.02, discretization='mac')
    assert row['status'] == 'complete'
    assert row['grid_cells'] == 16 and row['gauss_nodes'] is None
    assert row['max_physical_divergence'] < 1e-10
    assert row['production_target']['T'] == 100000.
    assert row['production_target']['field_interval'] == 10.
    independent = audit_mac(directory)
    assert independent['final_divergence_inf'] < 1e-10
    assert independent['energy'] == pytest.approx(row['final_observables']['energy'])
    with pytest.raises(ValueError, match='representation'):
        run_probe(tmp_path/'invalid', nodes=16, seed=directory/'final.npz',
                  duration=.02, discretization='spectral')

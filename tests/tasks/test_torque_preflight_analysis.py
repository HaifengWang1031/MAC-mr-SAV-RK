"""Analytic polynomial reference for the independent saved-field audit."""
import numpy as np
import pytest

from experiments.torque_transition.analyze_preflight import reconstruct


def test_independent_reconstruction_has_known_polynomial_integrals(tmp_path):
    # u=(P0-P2)(xi)*(P0-P2)(eta), v=0 on [0,2pi]^2.
    path = tmp_path/'state.npz'
    np.savez(path, velocity=[1., 0., 0., 0., 0., 0., 0., 0.], t=10.)
    result = reconstruct(path, nodes=16)
    assert result['energy'] == pytest.approx(.72, abs=1e-14)
    assert result['angular_momentum'] == pytest.approx(0., abs=1e-14)
    assert result['relative_divergence'] == pytest.approx(1/np.sqrt(2), abs=1e-14)
    assert result['divergence_rms'] == pytest.approx(np.sqrt(3.6)/np.pi, abs=1e-14)
    assert result['enstrophy'] == pytest.approx(1.8/np.pi**2, abs=1e-14)

"""Preflight configuration and source checks, independent of long-run claims."""
import numpy as np
import pytest

from experiments.torque_transition.preflight import make_model, modes_for_gauss, observables


def test_gauss_budget_preserves_polynomial_dealiasing():
    assert modes_for_gauss(128) == 84
    model = make_model(128, 0)
    assert model.space.size + model.quadrature_extra == 128
    assert 2*128-1 >= 3*(model.space.size+1)
    model.close()


@pytest.mark.parametrize('case, expected', [(0, 0.), (2, .001522409349774265), (12, .04)])
def test_force_torque_against_continuum_integral(case, expected):
    model = make_model(128, case)
    zero = model.state(0., model.zero_velocity())
    result = observables(model, zero, case)
    assert result['torque'] == pytest.approx(expected, abs=2e-14)
    assert result['energy'] == 0.
    assert result['angular_momentum'] == 0.
    force = model.force(0.)
    assert np.isfinite(force).all() and np.linalg.norm(force) > 0
    model.close()

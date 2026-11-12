"""Actual FEM workflow MMS force modes and default quadrature qualification."""
import numpy as np
import pytest

from experiments.fem_ns.problems import build_model, manufactured
from experiments.fem_ns.run import effective_config


@pytest.mark.parametrize('nx,ny,amplitude,expected_low_error', [
    (2, 2, 1., 1.4989270681157218e-5),
    (8, 4, .1, 2.299799689839733e-10),
], ids=['coarse-unit-amplitude', 'workflow-defaults'])
def test_workflow_mms_force_quadrature_sensitivity(nx, ny, amplitude, expected_low_error):
    models = {q: build_model(effective_config({'nx': nx, 'ny': ny, 'amplitude': amplitude,
                                             'quadrature_order': q}))[0] for q in (8, 16, 18)}
    projected = {q: model.force(0.) for q, model in models.items()}
    reference = models[18]
    errors = {}
    for q in (8, 16):
        np.testing.assert_array_equal(models[q].free_dofs, reference.free_dofs)
        difference = projected[q]-projected[18]
        errors[q] = np.sqrt(reference.inner(difference, difference))
    # Independent review measurements of the actual degree-15 force/test integrand.
    assert errors[8] == pytest.approx(expected_low_error, rel=.01, abs=1e-14)
    assert errors[16] < 2e-13


def test_known_mms_force_modes_match_continuous_reference_projection():
    cfg = effective_config({'nx': 6, 'ny': 4, 'T': .02, 'dt': .01})
    model, _ = build_model(cfg)
    _, _, force = manufactured(cfg['lx'], cfg['ly'], cfg['nu'], cfg['amplitude'])
    for time in (0., .17, .8, 2.1):
        before = model.performance_metrics()['projection_calls']
        actual = model.force(time)
        assert model.performance_metrics()['projection_calls'] == before
        expected = model.project(lambda x, y: force(x, y, time))
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

"""Consistent mass CG preserves the operator, full coefficient layout and SAV step."""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import null_space

from solver.fem import FEMModel, Space
from solver.fem.mass import VelocityMassFactor
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV


@pytest.mark.parametrize('backend', ['cg', 'metis'])
@pytest.mark.parametrize('coupled', [False, True])
def test_cg_mass_inverse_matches_independently_specified_spd_system(coupled: bool, backend: str):
    a = np.array([[2., 0., 1., 0.], [0., 2., 0., 1.],
                  [1., 0., 3., 0.], [0., 1., 0., 3.]])
    if coupled:
        a[0, 1] = a[1, 0] = .2
    inverse = VelocityMassFactor(sp.csc_matrix(a), [np.array([0, 2]), np.array([1, 3])], backend=backend)
    expected = np.array([[1., 0., -.3], [2., 1., .5], [3., -1., .7], [4., 2., -.9]])
    for values in (expected[:, 0], expected):
        rhs = a @ values
        before = rhs.copy();rhs.flags.writeable = False
        np.testing.assert_allclose(inverse.solve(rhs), values, atol=5e-13, rtol=0.)
        np.testing.assert_array_equal(rhs, before)
    if backend == 'cg':
        assert inverse.nnz == 0 and inverse.metrics()['cg_iterations'] > 0
    else:
        assert inverse.nnz > 0 and inverse.metrics()['ordering_wall_seconds'] >= 0.
    assert len(inverse.components) == (0 if coupled else 2)


@pytest.mark.parametrize('backend', ['cg', 'metis'])
def test_mass_cg_preserves_all_four_sav_stages_and_nonzero_scalar(backend: str):
    direct = FEMModel(Space(5, 4), .1)
    iterative = FEMModel(Space(5, 4), .1, mass_backend=backend)
    initial = .05*null_space(direct.divergence.toarray())[:, 0]
    results = [SDIRK3MRSAV().step(m, m.state(0., initial, .03), .002)
               for m in (direct, iterative)]
    np.testing.assert_allclose(results[1].state.u, results[0].state.u, atol=3e-12, rtol=0.)
    np.testing.assert_allclose(results[1].state.v, results[0].state.v, atol=3e-12, rtol=0.)
    assert results[1].state.r == pytest.approx(results[0].state.r, abs=3e-13)
    for a, b in zip(results[0].stages, results[1].stages):
        np.testing.assert_allclose(b.pressure, a.pressure, atol=2e-10, rtol=0.)
        assert b.scalar_residual < 1e-10 and b.continuity_residual < 1e-10
    assert (iterative.performance_metrics()['mass_factor_nonzeros'] == 0) == (backend=='cg')

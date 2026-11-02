"""Weak momentum preserves the equations, traces and SAV roots of all six formats."""
import numpy as np
import pytest
from skfem import MeshTri

from solver.fem import FEMModel, MeshSpace
from solver.fem.cavity import ClosedCavityModel
from solver.fem.hele_shaw import HeleShawModel
from solver.fem.open_flow import OpenFlowModel
from solver.momentum import stage_momentum
from solver.schemes.registry import make_scheme


def model(family, **options):
    mesh = MeshTri.init_tensor(np.linspace(0., 2., 7), np.linspace(0., 1., 5)).with_boundaries({
        'lid': lambda x: np.isclose(x[1], 1.),
        'walls': lambda x: np.isclose(x[1], 0.) | np.isclose(x[1], 1.),
        'inlet': lambda x: np.isclose(x[0], 0.), 'outlet': lambda x: np.isclose(x[0], 2.)})
    space = MeshSpace(mesh)
    if family == 'closed':
        return FEMModel(space, .1, **options)
    if family == 'lid':
        return ClosedCavityModel(space, .1, **options)
    if family == 'hele':
        return HeleShawModel(space, .1, .4, pressure_drop=1., inlet_treatment='energy_stable_pressure',
                             outlet_treatment='backflow_damped', **options)
    return OpenFlowModel(space, .1, dirichlet_facets=np.concatenate([
        mesh.boundaries['walls'], mesh.boundaries['inlet']]),
        boundary_velocity=lambda x, y: (np.where(np.isclose(x, 0.), y*(1-y), 0.), 0.),
        outlet_treatment='backflow_damped', **options)


@pytest.mark.parametrize('family', ['closed', 'lid', 'open', 'hele'])
@pytest.mark.parametrize('scheme', ['imex_euler', 'imex_euler_mrsav', 'sdirk2', 'sdirk2_mrsav',
                                   'sdirk3', 'sdirk3_mrsav'])
def test_all_formats_keep_physical_fields_pressure_roots_and_reduce_mass_solves(family, scheme):
    direct, weak = model(family), model(family, fem_load_mode='weak', mass_backend='cg')
    rng = np.random.default_rng(51)
    velocity = direct.solve(.02*rng.normal(size=len(direct.free_dofs)), mass=1., viscosity=0.).velocity
    states = [m.state(.1, velocity, .03) for m in (direct, weak)]
    before = [m.performance_metrics()['mass_solves'] for m in (direct, weak)]
    results = [make_scheme(scheme).step(m, state, .0005) for m, state in zip((direct, weak), states)]
    np.testing.assert_allclose(results[1].state.u, results[0].state.u, atol=2e-10, rtol=0.)
    np.testing.assert_allclose(results[1].state.v, results[0].state.v, atol=2e-10, rtol=0.)
    assert results[1].state.r == pytest.approx(results[0].state.r, abs=2e-11)
    for a, b in zip(results[0].stages, results[1].stages):
        np.testing.assert_allclose(b.pressure, a.pressure, atol=2e-8, rtol=0.)
        np.testing.assert_allclose(b.candidates, a.candidates, atol=2e-9, rtol=0.)
        assert b.residual < 1e-8 and b.continuity_residual < 1e-9
        assert b.scalar_residual < 1e-8
    after = [m.performance_metrics()['mass_solves'] for m in (direct, weak)]
    assert after[1]-before[1] < after[0]-before[0]


def test_load_normalization_is_a_lower_bound_and_defect_uses_consistent_inverse():
    m = model('closed', fem_load_mode='weak')
    rng = np.random.default_rng(8)
    load = rng.normal(size=(len(m.free_dofs), 3))
    exact = np.linalg.solve(m.mass.toarray(), load)
    lower = m.stokes.load_norm_lower_bound(load)
    assert np.all(lower <= np.max(abs(exact), axis=0))
    momentum = stage_momentum(m)
    defect = load[:, 0]*1e-6
    expected = np.max(abs(exact[:, 0]))*1e-6/(1+lower[1]+.02*lower[2])
    assert momentum.residual(defect, load[:, 1], load[:, 2], .02) == pytest.approx(expected, rel=1e-12)
    assert momentum.residual(defect, load[:, 1], load[:, 2], .02) >= \
        np.max(abs(exact[:, 0]))*1e-6/(1+np.max(abs(exact[:, 1]))+.02*np.max(abs(exact[:, 2])))

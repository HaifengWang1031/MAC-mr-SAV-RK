"""Manufactured exact mixed solutions exercise all B rows and pressure constraints."""
import importlib.util

import numpy as np
import pytest
from scipy.linalg import null_space

from solver.fem import FEMModel, Space
from solver.fem.multigrid import PETScStokes

pytestmark = pytest.mark.skipif(importlib.util.find_spec('petsc4py') is None,
                                reason='optional matching PETSc/petsc4py')


def test_declared_krylov_convergence_is_refined_against_the_exact_mixed_equations(monkeypatch):
    m = FEMModel(Space(5, 4), .1, stokes_backend='petsc')
    velocity = .01*np.sin(np.arange(len(m.free_dofs)))
    pressure = m.pressure_basis.doflocs[0].copy()
    pressure -= (m.pressure_weights @ pressure)/sum(m.pressure_weights)
    load = m.mass @ velocity+.003*(m.linear_stiffness @ velocity)-m.divergence.T @ pressure
    real_solve = m.stokes._multigrid.solve
    calls = 0
    def perturbed_first_solve(*args, **kwargs):
        nonlocal calls
        value = real_solve(*args, **kwargs)
        if calls == 0:value[0, 0] += 1e-6
        calls += 1
        return value
    monkeypatch.setattr(m.stokes._multigrid, 'solve', perturbed_first_solve)
    try:
        result = m.stokes.solve_load_columns([load], mass=1., viscosity=.003,
                                             divergence_rhs=(m.divergence @ velocity)[:, None])[0]
        np.testing.assert_allclose(result.velocity, velocity, atol=1e-10, rtol=0.)
        np.testing.assert_allclose(result.pressure, pressure, atol=1e-10, rtol=0.)
        assert m.stokes.metrics()['petsc_refinements'] == 1
        assert m.stokes.metrics()['petsc_fallbacks'] == 0
        assert m.stokes.metrics()['factorizations'] == 0
    finally:
        m.stokes.close()


@pytest.mark.parametrize('gauge', [False, True])
def test_block_amg_solves_known_two_rhs_with_changing_shift_and_constraint(gauge: bool):
    from skfem import MeshTri
    from solver.fem import MeshSpace
    mesh = MeshTri.init_tensor(np.linspace(0., 2., 7), np.linspace(0., 1., 5))
    facets = mesh.facets_satisfying(lambda x: np.isclose(x[1], 0.) | np.isclose(x[1], 1.))
    m = FEMModel(MeshSpace(mesh), .1, pressure_gauge=gauge,
                 dirichlet_facets=None if gauge else facets, reaction_coefficient=7.)
    parts = [np.flatnonzero(np.isin(m.free_dofs, ids)) for ids in m.component_dofs]
    solver = PETScStokes(m.mass, m.linear_stiffness, m.divergence, m.pressure_weights,
        pressure_gauge=gauge, reaction_coefficient=7., components=parts)
    null = null_space(m.divergence.toarray())
    p = m.pressure_basis.doflocs[0]+2*m.pressure_basis.doflocs[1]
    if gauge:
        p = p-float(m.pressure_weights @ p)/np.sum(m.pressure_weights)
    expected = np.zeros((len(m.free_dofs)+len(p)+int(gauge), 3))
    expected[:len(m.free_dofs), 0] = null[:, 0]
    expected[:len(m.free_dofs), 2] = null[:, 1]
    expected[len(m.free_dofs):len(m.free_dofs)+len(p), 0] = p
    expected[len(m.free_dofs):len(m.free_dofs)+len(p), 2] = -p
    try:
        for mass, viscosity in [(1., 0.), (1., .003), (1., .00303), (0., .1)]:
            matrix = m.stokes._matrix((mass, viscosity))
            rhs = matrix @ expected
            result = solver.solve(matrix, rhs, mass=mass, viscosity=viscosity)
            np.testing.assert_allclose(result, expected, atol=2e-8, rtol=0.)
            assert np.max(abs(matrix @ result-rhs)) < 1e-9
            assert np.max(abs(m.divergence @ result[:len(m.free_dofs)])) < 1e-9
        assert solver.metrics()['iterations'] > 0
        assert solver.metrics()['velocity_setups'] == 3
    finally:
        solver.close()


@pytest.mark.parametrize('load_mode', ['riesz', 'weak'])
@pytest.mark.parametrize('open_flow', [False, True])
def test_public_sav_steps_keep_lifting_reaction_and_physical_pressure(open_flow: bool, load_mode: str):
    from skfem import MeshTri
    from solver.fem import MeshSpace
    from solver.fem.cavity import ClosedCavityModel
    from solver.fem.hele_shaw import HeleShawModel
    from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV
    mesh = MeshTri.init_tensor(np.linspace(0., 2., 8), np.linspace(0., 1., 6)).with_boundaries({
        'lid': lambda x: np.isclose(x[1], 1.),
        'walls': lambda x: np.isclose(x[1], 0.) | np.isclose(x[1], 1.),
        'inlet': lambda x: np.isclose(x[0], 0.), 'outlet': lambda x: np.isclose(x[0], 2.)})
    options = [{}, dict(stokes_backend='petsc', mass_backend='cg', fem_load_mode=load_mode)]
    if open_flow:
        models = [HeleShawModel(MeshSpace(mesh), .1, .4, pressure_drop=1.,
            inlet_treatment='energy_stable_pressure', outlet_treatment='backflow_damped', **kw) for kw in options]
    else:
        models = [ClosedCavityModel(MeshSpace(mesh), .1, **kw) for kw in options]
    try:
        results = [SDIRK3MRSAV().step(m, m.state(0., m.zero_velocity(), .02), .002) for m in models]
        np.testing.assert_allclose(results[1].state.u, results[0].state.u, atol=2e-10, rtol=0.)
        np.testing.assert_allclose(results[1].state.v, results[0].state.v, atol=2e-10, rtol=0.)
        assert results[1].state.r == pytest.approx(results[0].state.r, abs=2e-11)
        for direct, mg in zip(results[0].stages, results[1].stages):
            np.testing.assert_allclose(mg.pressure, direct.pressure, atol=2e-9, rtol=0.)
            assert mg.continuity_residual < 1e-9 and mg.scalar_residual < 1e-9
        if open_flow:
            pressures = [m.instantaneous_solution(m.vector(r.state), r.state.t)[1] for m, r in zip(models, results)]
            np.testing.assert_allclose(pressures[1], pressures[0], atol=2e-9, rtol=0.)
            assert models[1].performance_metrics()['pressure_petsc_checked_solves'] == 1
        metrics = models[1].performance_metrics()
        assert metrics['stokes_petsc_checked_solves'] == 4
        assert metrics['stokes_petsc_fallbacks'] == 0 and metrics['stokes_factorizations'] == 0
    finally:
        for m in models:
            m.stokes.close()
            if hasattr(m, '_pressure_stokes'):
                m._pressure_stokes.close()


@pytest.mark.parametrize('family', ['fem', 'cylinder', 'wedge', 'hele_shaw'])
def test_public_run_load_keeps_explicit_backend_identity_without_analysis_solver(tmp_path, monkeypatch, family: str):
    from experiments.fem_ns.run import run as fem_run, load_fem
    from experiments.cylinder_ns.run import run as cylinder_run, load_cylinder
    from experiments.wedge_cavity.run import run as wedge_run
    from experiments.wedge_cavity.report import open_run
    from experiments.hele_shaw_airfoil.run import run as hele_run
    from experiments.hele_shaw_airfoil.records import load as hele_load
    from solver.fem import multigrid
    mass_backend = 'metis' if family=='wedge' else 'cg'
    options = dict(stokes_backend='petsc', mass_backend=mass_backend, stokes_amg_type='gamg', fem_load_mode='weak',
                   T=.004, dt=.002, scheme='sdirk3_mrsav')
    if family == 'fem':
        directory = fem_run({**options, 'nx': 4, 'ny': 3}, root=tmp_path)
        reader = lambda: load_fem(directory, require_complete=True)
    elif family == 'cylinder':
        directory = cylinder_run({**options, 'resolution': .25, 'mode': 'fixed'}, root=tmp_path)
        reader = lambda: load_cylinder(directory, require_complete=True)
    elif family == 'wedge':
        directory = wedge_run({**options, 'lid_cells': 8, 'snapshot_every': .002}, root=tmp_path)
        reader = lambda: open_run(directory)
    else:
        pytest.importorskip('gmsh')
        directory = hele_run({**options, 'gap_ratio': .3, 'layer_cells': 1,
            'surface_points': 32, 'far_size': .6, 'darcy_reference': False,
            'inlet_treatment': 'energy_stable_pressure'}, root=tmp_path)
        reader = lambda: hele_load(directory)
    def forbidden():raise AssertionError('Loading an existing record constructed a PETSc solver')
    monkeypatch.setattr(multigrid, '_petsc', forbidden)
    cfg, manifest, *_ = reader()
    assert cfg['fem_load_mode'] == 'weak'
    assert cfg['mass_backend'] == mass_backend and cfg['stokes_amg_type'] == 'gamg'
    assert cfg['stokes_backend'] == 'petsc' and manifest['status'] == 'complete'
    assert manifest['source']['petsc']['communicator'] == 'COMM_SELF'
    assert manifest['source']['packages']['petsc4py']

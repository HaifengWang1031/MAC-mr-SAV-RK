"""Independent analytic references for Darcy, reaction, pressure work and SAV."""
import numpy as np
import pytest
from skfem import MeshTri

from solver.fem import MeshSpace, OpenFlowModel
from solver.fem.darcy import solve_darcy
from solver.fem.hele_shaw import HeleShawModel
from solver.schemes.registry import SCHEMES, make_scheme


def channel(ny=12):
    return MeshTri.init_tensor(np.linspace(0., 2., 7), np.linspace(-.5, .5, ny+1)).with_boundaries({
        'inlet': lambda x: np.isclose(x[0], 0.), 'outlet': lambda x: np.isclose(x[0], 2.),
        'walls': lambda x: np.isclose(np.abs(x[1]), .5)})


def test_darcy_pressure_and_boundary_flux_are_known_exactly():
    mesh = channel(4)
    result = solve_darcy(mesh, .1, .2, 3.)
    from skfem import Basis, ElementTriP2
    basis = Basis(mesh, ElementTriP2())
    np.testing.assert_allclose(result.pressure, 3.*(1-basis.doflocs[0]/2), atol=4e-13)
    speed = .2**2/(12*.1)*1.5
    np.testing.assert_allclose(result.u, speed, atol=2e-13)
    np.testing.assert_allclose(result.v, 0., atol=2e-13)
    assert result.fluxes['inlet'] == pytest.approx(-speed, abs=2e-13)
    assert result.fluxes['outlet'] == pytest.approx(speed, abs=2e-13)
    assert abs(sum(result.fluxes.values())) < 2e-13
    assert result.residual < 1e-12


@pytest.mark.parametrize('scheme', SCHEMES)
def test_reaction_includes_nonzero_trace_mass_load_and_all_shared_stages(scheme):
    mesh = channel(4)
    fixed = np.concatenate([mesh.boundaries[name] for name in ('inlet', 'walls')])
    nu, kappa = .1, 13.
    model = OpenFlowModel(MeshSpace(mesh), nu, dirichlet_facets=fixed,
                          boundary_velocity=lambda x, y: (.25-y*y, 0.), reaction_coefficient=kappa)
    forcing = nu*kappa*model.project(lambda x, y: (.25-y*y, 0.))
    model.force = lambda t: forcing.copy()
    solved = model.solve(forcing, mass=0., viscosity=nu)
    points = np.array([[.2, .7, 1.8], [-.2, .1, .3]])
    expected = np.array([.25-points[1]**2, np.zeros(3)])
    np.testing.assert_allclose(model.evaluate_velocity(solved.velocity, points), expected, atol=2e-12)
    np.testing.assert_allclose(model.evaluate_pressure(solved.pressure, points), 2*nu*(2-points[0]), atol=3e-12)
    trial = make_scheme(scheme).step(model, model.state(0., solved.velocity), .007)
    np.testing.assert_allclose(model.vector(trial.state), solved.velocity, atol=4e-11)
    assert abs(trial.state.r) < 1e-10
    assert max(s.continuity_residual for s in trial.stages) < 1e-11


def test_linear_brinkman_profile_pressure_scale_and_energy_balance():
    errors, pressure_errors = [], []
    for ny in [8, 16]:
        model = HeleShawModel(MeshSpace(channel(ny)), .1, .4, pressure_drop=1.2, convection=False)
        solved = model.solve(model.force(0.), mass=0., viscosity=model.nu)
        x = model.velocity_basis.global_coordinates()
        velocity = model.velocity_basis.interpolate(model.full_velocity(solved.velocity))
        length = np.sqrt(model.nu/model.drag)
        gradient = 1.2/(1.2*2.)
        exact = gradient/model.drag*(1.-np.cosh(x[1]/length)/np.cosh(.5/length))
        errors.append(np.sqrt(np.sum((velocity[0]-exact)**2*model.velocity_basis.dx)))
        physical = model.physical_pressure(solved.pressure)
        # The cosh velocity is not in P2; its error perturbs even the P1 pressure.
        pressure_errors.append(np.max(np.abs(physical-1.2*(1-model.pressure_basis.doflocs[0]/2))))
        state = model.state(0., solved.velocity)
        d = model.diagnostics(state)
        assert abs(d['pressure_power']-d['gap_dissipation']-d['viscous_dissipation']) < 2e-12
        assert abs(d['boundary_flux']) < 2e-12
    assert errors[1] < errors[0]/5.
    assert pressure_errors[1] < pressure_errors[0]/3.
    assert pressure_errors[0] < .002


def test_inertial_sav_uses_scaled_open_boundary_work_and_only_damps_outlet():
    model = HeleShawModel(MeshSpace(channel(5)), .1, .4, pressure_drop=1.,
                          outlet_treatment='backflow_damped', ramp_time=.02)
    assert model.imposed_pressure(0.) == 0. and model.imposed_pressure(.02) == 1.
    velocity = np.random.default_rng(8).normal(size=len(model.free_dofs))*.01
    nonlinear, work = model.nonlinear_with_lifting(velocity)
    assert abs(model.inner(nonlinear, velocity)+work) < 2e-14
    assert np.array_equal(np.flatnonzero(model._open_mask[:, 0]),
                          np.flatnonzero(np.isin(model._all_facets.find, model.velocity_basis.mesh.boundaries['outlet'])))
    initial = model.state(0., model.zero_velocity())
    for scheme in ['sdirk2_mrsav', 'sdirk3_mrsav']:
        trial = make_scheme(scheme).step(model, initial, .002)
        assert np.isfinite(trial.state.u).all()
        assert max(s.residual for s in trial.stages) < 1e-10


def test_gmsh_airfoil_has_one_hole_and_saved_boundary_topology():
    pytest.importorskip('gmsh')
    from solver.fem.airfoil import make_airfoil_mesh, airfoil_polygon, validate_airfoil_mesh
    mesh = make_airfoil_mesh(gap_ratio=.3, layer_cells=1, surface_points=32, far_size=.6)
    result = validate_airfoil_mesh(mesh, airfoil_polygon(surface_points=32), (-3., 6., -2., 2.))
    assert result['area_error'] < 1e-10
    assert len(mesh.boundaries['airfoil']) >= 32


@pytest.mark.parametrize('scheme,order', [(s, 1 if s.startswith('imex') else 3 if s.startswith('sdirk3') else 2)
                                        for s in SCHEMES])
def test_time_refinement_of_forced_inertial_brinkman_with_known_exact_state(scheme, order):
    from solver.integrate import integrate, step_sizes
    model = HeleShawModel(MeshSpace(channel(3)), .1, .5, pressure_drop=1., convection=False)
    drive = model.force(0.)
    steady = model.solve(drive, mass=0., viscosity=model.nu).velocity
    model.beta = 9/7
    nonlinear = model.nonlinear(steady)
    # Exact semidiscrete solution u(t)=(1-exp(-t))*steady, p(t) similarly scaled.
    # Homogeneity N(s*u)=s^2*N(u), and the steady linear solve fixes L*u+G*p.
    model.force = lambda t: np.exp(-t)*steady+(1-np.exp(-t))*drive+(1-np.exp(-t))**2*nonlinear
    exact = (1-np.exp(-.4))*steady
    errors = []
    for dt in [.05, .025, .0125]:
        result = integrate(model, make_scheme(scheme), model.state(0., model.zero_velocity()), step_sizes(.4, dt=dt))
        assert result.status == 'complete'
        error = model.vector(result.final)-exact
        errors.append(np.sqrt(model.inner(error, error)))
    assert errors[2] < errors[1]/2**(order-.35), (scheme, errors)

"""Tagged cylinder geometry and independently known hydrostatic force at Model seam."""
import numpy as np
import pytest

from solver.fem import CylinderSpace, OpenFlowModel


def test_cylinder_hole_and_boundary_geometry_are_not_a_filled_rectangle():
    space = CylinderSpace(resolution=.25, circle_segments=32)
    velocity, pressure = space.bases()
    mesh = velocity.mesh
    assert mesh.boundaries is not None
    assert set(mesh.boundaries) == {'inlet', 'outlet', 'walls', 'cylinder'}
    expected_area = 2.2*.41-.5*32*.05**2*np.sin(2*np.pi/32)
    assert np.sum(velocity.dx) == pytest.approx(expected_area, abs=2e-13)
    assert len(mesh.boundaries['cylinder']) == 32
    circle_nodes = np.unique(mesh.facets[:, mesh.boundaries['cylinder']])
    np.testing.assert_allclose(np.linalg.norm(mesh.p[:, circle_nodes]-np.array([[.2], [.2]]), axis=0),
                               .05, atol=2e-15)
    assert pressure.N == mesh.p.shape[1]
    finer = CylinderSpace(resolution=.5, circle_segments=64)
    assert finer.mesh.t.shape[1] > mesh.t.shape[1]
    assert finer.sagitta < space.sagitta/3.9
    assert mesh.p.shape[1]-mesh.facets.shape[1]+mesh.t.shape[1] == 0


def test_default_mesh_resolves_the_re100_wall_layer_without_global_overrefinement():
    space = CylinderSpace(1.)
    assert space.circle_segments == 96
    radius = np.linalg.norm(space.mesh.p-np.array([[.2], [.2]]), axis=0)
    first_layer = np.min(radius[radius > .05+1e-10])-.05
    assert first_layer <= .0025+1e-12
    assert space.mesh.t.shape[1] < 8000


def test_linear_hydrostatic_pressure_has_known_polygon_obstacle_force():
    space = CylinderSpace(resolution=.25, circle_segments=32)
    assert space.mesh.boundaries is not None
    fixed = np.concatenate([space.mesh.boundaries[name] for name in ('inlet', 'walls', 'cylinder')])
    model = OpenFlowModel(space, .01, dirichlet_facets=fixed,
                          boundary_velocity=lambda x, y: (0., 0.))
    force = model.project(lambda x, y: (1., 0.))
    model.force = lambda t: force
    solved = model.solve(force, mass=0., viscosity=model.nu)
    points = np.array([[.1, .15, .25, 2.1], [.1, .199999, .199999, .3]])
    np.testing.assert_allclose(model.evaluate_velocity(solved.velocity, points), 0., atol=2e-11)
    np.testing.assert_allclose(model.evaluate_pressure(solved.pressure, points), points[0]-2.2, atol=3e-11)
    area = .5*32*.05**2*np.sin(2*np.pi/32)
    np.testing.assert_allclose(model.boundary_force(solved.velocity, solved.pressure, 'cylinder'),
                               [-area, 0.], atol=5e-12)
    rate, pressure = model.instantaneous_solution(solved.velocity, 0.)
    np.testing.assert_allclose(rate, 0., atol=5e-11)
    np.testing.assert_allclose(model.evaluate_pressure(pressure, points), points[0]-2.2, atol=5e-11)
    assert abs(model.pressure_integral(pressure)) > .5
    assert model.max_abs(model.apply_D(solved.velocity)) < 1e-12
    with pytest.raises(ValueError):
        model.evaluate_velocity(solved.velocity, np.array([[.2], [.2]]))


def test_quadratic_shear_has_known_viscous_obstacle_force_and_gradient_orientation():
    space = CylinderSpace(resolution=.25, circle_segments=32)
    assert space.mesh.boundaries is not None
    fixed = np.concatenate([space.mesh.boundaries[name] for name in ('inlet', 'walls', 'cylinder')])
    nu = .3
    model = OpenFlowModel(space, nu, dirichlet_facets=fixed, boundary_velocity=lambda x, y: (0., x*x))
    full = np.zeros(model.velocity_basis.N)
    indices = model.component_dofs[1]
    full[indices] = model.velocity_basis.doflocs[0, indices]**2
    area = .5*32*.05**2*np.sin(2*np.pi/32)
    # U=(0,x²): integral_solid nu*grad(U)*n = integral_solid nu*Lap(U) = (0,2nu*area).
    np.testing.assert_allclose(model.boundary_force(full[model.free_dofs], model.zero_pressure(), 'cylinder'),
                               [0., 2*nu*area], atol=8e-13)

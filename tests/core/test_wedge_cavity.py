"""Moving-lid FEM corner trace and independent SAV lifting work."""
import numpy as np

from solver.fem.cavity import ClosedCavityModel
from solver.fem.wedge import WedgeSpace


def test_stationary_inclined_walls_include_both_lid_corners():
    model = ClosedCavityModel(WedgeSpace(apex_angle=25., lid_cells=12), .01)
    mesh, basis = model.velocity_basis.mesh, model.velocity_basis
    walls = np.concatenate([mesh.boundaries[key] for key in ('wall_left', 'wall_right')])
    fixed = basis.get_dofs(facets=walls).all()
    np.testing.assert_array_equal(model._boundary_values[fixed], 0.)
    full = model.full_velocity(model.zero_velocity())
    u_dofs = model.component_dofs[0]
    x, y = basis.doflocs[:, u_dofs]
    moving = (np.abs(y) < 1e-14) & (np.abs(x) < .5-1e-14)
    np.testing.assert_array_equal(full[u_dofs[moving]], 1.)
    np.testing.assert_array_equal(full[model.component_dofs[1]], 0.)


def test_lifting_work_matches_independent_skew_form_and_cancels_free_pairing():
    model = ClosedCavityModel(WedgeSpace(lid_cells=12), .01)
    velocity = np.random.default_rng(53).normal(scale=.15, size=len(model.free_dofs))
    full = model.full_velocity(velocity)
    physical = model.velocity_basis.interpolate(full)
    lifting = model.velocity_basis.interpolate(model._boundary_values)
    advected = np.einsum('ijeq,jeq->ieq', physical.grad, physical)
    test_advected = np.einsum('ijeq,jeq->ieq', lifting.grad, physical)
    reference = .5*float(np.sum((np.sum(advected*lifting, axis=0)
                                - np.sum(test_advected*physical, axis=0))*model.velocity_basis.dx))
    nonlinear, work = model.nonlinear_with_lifting(velocity)
    assert abs(reference) > 1e-6
    np.testing.assert_allclose(work, reference, atol=2e-14)
    np.testing.assert_allclose(model.inner(nonlinear, velocity)+work, 0., atol=2e-13)

"""Steady parabolic inlet, natural outlet and compatible Stokes startup."""
import numpy as np

from solver.core import State
from solver.fem import CylinderSpace, OpenFlowModel
from solver.fem.cylinder import HEIGHT
from solver.fem.space import Array


def inlet_velocity(x: Array, y: Array, maximum: float) -> tuple[Array, float]:
    return np.where(np.isclose(x, 0., rtol=0., atol=1e-12),
                    4*maximum*y*(HEIGHT-y)/HEIGHT**2, 0.), 0.


def build_model(cfg: dict) -> tuple[OpenFlowModel, State]:
    space = CylinderSpace(cfg['resolution'], cfg['circle_segments'], cfg['quadrature_order'])
    assert space.mesh.boundaries is not None
    fixed = np.concatenate([space.mesh.boundaries[k] for k in ('inlet', 'walls', 'cylinder')])
    model = OpenFlowModel(space, cfg['nu'], dirichlet_facets=fixed,
                          boundary_velocity=lambda x, y: inlet_velocity(x, y, cfg['inlet_max']),
                          tolerance=cfg['stokes_tolerance'], cache_size=cfg['cache_size'],
                          stokes_backend=cfg['stokes_backend'])
    steady = model.solve(model.zero_velocity(), mass=0., viscosity=model.nu)
    return model, model.state(0., steady.velocity, cfg['r0'])

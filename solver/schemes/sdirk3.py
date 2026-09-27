"""Four-stage, third-order incremental IMEX-SDIRK scheme from Obsidian note 02."""
from typing import Any
import numpy as np
from ..core import Trial
from ..model import Model, stage_divergences

D = np.array([[1/4, 0, 0, 0], [3/10, 1/4, 0, 0],
              [-31/220, 1/11, 1/4, 0], [31/198, 5/132, -4/9, 1/4]])
DHAT = np.array([[1/4, 0, 0, 0], [-33/50, 121/100, 0, 0],
                 [239/400, -71/100, 5/16, 0],
                 [-1/48, -17/66, 235/528, -1/6]])
C = np.array([0., 1/4, 4/5, 1., 1.])


def embedded_velocity(model: Model, velocities: list[Any]) -> Any:
    """Second-order stage combination; velocities are the node and four stages."""
    return model.combine((2/3, velocities[0]), (-128/99, velocities[1]),
                         (50/33, velocities[2]), (1/9, velocities[3]))


def known_terms(model: Model, nonlinear: list[tuple[Any, float]], forces: list[Any], i: int
                ) -> tuple[Any, Any, float]:
    """Incremental explicit force/convection and lifting pairing for stage i (0-based)."""
    convection = model.combine(*[(DHAT[i, j], nonlinear[j][0]) for j in range(i+1)])
    lifting_work = float(sum(DHAT[i, j]*nonlinear[j][1] for j in range(i+1)))
    force = model.combine(*[(DHAT[i, j], forces[j]) for j in range(i+1)])
    return convection, force, lifting_work


class SDIRK3:
    name = 'sdirk3'

    def step(self, model: Model, state: Any, dt: float) -> Trial:
        if not np.isfinite(dt) or dt <= 0: raise ValueError('Invalid step size')
        velocities = [model.vector(state)]
        linear: list[Any] = []
        stages = []
        nonlinear = [model.nonlinear_with_lifting(velocities[0])]
        forces = [model.force(state.t+C[j]*dt) for j in range(4)]
        for i in range(4):
            convection, force, _ = known_terms(model, nonlinear, forces, i)
            rhs = model.combine((1., velocities[-1]), (dt, force), (-dt, convection),
                                *[(-model.nu*dt*D[i,j], linear[j])
                                  for j in range(i)])
            solved = model.solve(rhs, mass=1., viscosity=model.nu*dt*D[i,i])
            velocities.append(solved.velocity)
            if i < 3:
                linear.append(model.apply_K(solved.velocity))
                nonlinear.append(model.nonlinear_with_lifting(solved.velocity))
            physical_divergence,continuity=stage_divergences(model,solved.velocity)
            stages.append(model.stage(model.combine((1/dt, solved.pressure)), solved.residual,
                                      physical_divergence,
                                      continuity_residual=continuity))
        return Trial(model.state(state.t+dt, velocities[-1]), stages,
                     embedded_velocity(model, velocities))

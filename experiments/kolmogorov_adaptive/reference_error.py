"""Online, grid-aligned velocity reference errors for natural adaptive nodes."""
from dataclasses import dataclass, replace
from math import ceil
from time import process_time

import numpy as np

from solver.core import State
from solver.mac.grid import MACGrid
from solver.schemes.sdirk3 import SDIRK3


@dataclass
class ReferenceTrack:
    """A fine SDIRK3 trajectory independently advanced to accepted nodes."""
    model: object
    state: State
    max_step: float
    min_substeps: int
    steps: int = 0
    cpu_seconds: float = 0.

    def advance_to(self, target: float) -> State:
        gap = target-self.state.t
        if not np.isfinite(gap) or gap <= 0:
            raise ValueError('Reference target must be a later finite time')
        parts = max(self.min_substeps, ceil(gap/self.max_step))
        step = gap/parts
        start = process_time()
        scheme = SDIRK3()
        for _ in range(parts):
            trial = scheme.step(self.model, self.state, step)
            self.state = trial.state
            if not np.isfinite(self.model.vector(self.state)).all():
                raise FloatingPointError('Nonfinite reference velocity')
        self.cpu_seconds += process_time()-start
        self.steps += parts
        if abs(self.state.t-target) > 1e-10*max(1., abs(target)):
            raise RuntimeError('Reference did not arrive at adaptive time')
        # Remove only accumulated time-label roundoff before the next interval.
        self.state = replace(self.state, t=target)
        return self.state


class ReferenceErrorObserver:
    """Compare one adaptive trajectory against two nested reference resolutions."""

    def __init__(self, grid: MACGrid, coarse_model: object, coarse_initial: State,
                 fine_model: object, fine_initial: State, *, max_step: float,
                 min_substeps: int = 8):
        if not np.isfinite(max_step) or max_step <= 0 or min_substeps < 2:
            raise ValueError('Invalid reference refinement controls')
        self.grid = grid
        self.coarse = ReferenceTrack(coarse_model, coarse_initial, max_step,
                                     min_substeps)
        self.fine = ReferenceTrack(fine_model, fine_initial, max_step/2,
                                   2*min_substeps)
        self.times = [coarse_initial.t]
        self.velocity_error = [0.]
        self.reference_difference = [0.]

    def __call__(self, adaptive_state: State) -> None:
        coarse = self.coarse.advance_to(adaptive_state.t)
        fine = self.fine.advance_to(adaptive_state.t)
        pack = self.grid.pack
        norm = self.grid.norm
        coarse_velocity = pack(coarse.u, coarse.v)
        fine_velocity = pack(fine.u, fine.v)
        adaptive_velocity = pack(adaptive_state.u, adaptive_state.v)
        scale = max(norm(fine_velocity), 1e-14)
        self.times.append(adaptive_state.t)
        self.velocity_error.append(norm(adaptive_velocity-fine_velocity)/scale)
        self.reference_difference.append(norm(coarse_velocity-fine_velocity)/scale)

    def summary(self, verification_fraction: float) -> dict:
        errors = np.asarray(self.velocity_error)
        differences = np.asarray(self.reference_difference)
        max_error = float(errors.max())
        final_error = float(errors[-1])
        max_difference = float(differences.max())
        final_difference = float(differences[-1])
        verified = (max_difference <= verification_fraction*max_error and
                    final_difference <= verification_fraction*final_error)
        return {
            'max_relative_l2': max_error,
            'final_relative_l2': final_error,
            'max_reference_refinement_difference': max_difference,
            'final_reference_refinement_difference': final_difference,
            'reference_verified': bool(verified),
            'verification_fraction': verification_fraction,
            'coarse_reference_steps': self.coarse.steps,
            'fine_reference_steps': self.fine.steps,
            'reference_cpu_seconds': self.coarse.cpu_seconds+self.fine.cpu_seconds,
        }

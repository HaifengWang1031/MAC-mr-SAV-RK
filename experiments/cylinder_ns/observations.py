"""Accepted-state force/probe measurements, separate from stage pressures."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from solver.core import State
from solver.fem import OpenFlowModel
from solver.fem.cylinder import RADIUS
from solver.fem.space import Array

PROBES = np.array([[.15, .25], [.199999, .199999]])
FIELDS = ('time', 'drag', 'lift', 'pressure_difference', 'pressure_front', 'pressure_rear', 'force_x', 'force_y')


@dataclass
class Observations:
    model: OpenFlowModel
    inlet_max: float
    rows: list[dict[str, float]] = field(default_factory=list)
    pressure: Array | None = None
    pressure_time: float | None = None
    _probes: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._probes = self.model.pressure_basis.probes(PROBES)

    def __call__(self, state: State) -> None:
        velocity = self.model.vector(state)
        _, pressure = self.model.instantaneous_solution(velocity, state.t)
        force = self.model.boundary_force(velocity, pressure, 'cylinder')
        probes = self._probes @ pressure
        factor = 2/((2*self.inlet_max/3)**2*(2*RADIUS))
        row = {'time': float(state.t), 'drag': float(factor*force[0]), 'lift': float(factor*force[1]),
               'pressure_difference': float(probes[0]-probes[1]),
               'pressure_front': float(probes[0]), 'pressure_rear': float(probes[1]),
               'force_x': float(force[0]), 'force_y': float(force[1])}
        if not np.isfinite(list(row.values())).all():
            raise FloatingPointError('Nonfinite cylinder force/pressure observation')
        self.rows.append(row)
        self.pressure, self.pressure_time = pressure, float(state.t)

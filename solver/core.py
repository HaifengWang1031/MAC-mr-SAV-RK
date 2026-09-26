"""Accepted state and stage diagnostics; schemes do not mutate their input."""
from dataclasses import dataclass, field
from typing import Any, Protocol
from .mac.grid import Array
from .model import Model

@dataclass(frozen=True)
class State:
    t: float
    u: Array
    v: Array
    r: float = 0.0

@dataclass
class Stage:
    pressure: Array
    residual: float
    divergence_inf: float
    r: float = 0.0
    candidates: list[float] = field(default_factory=list)
    root_residuals: list[float] = field(default_factory=list)
    scalar_residual: float = 0.0
    continuity_residual: float = float("nan")

@dataclass
class Trial:
    state: State
    stages: list[Stage]
    embedded_velocity: Any | None = None

@dataclass
class History:
    """Only the most recent accepted state is needed by these one-step schemes."""
    state: State
    accepted_steps: int = 0

class Scheme(Protocol):
    name: str
    # `state` is Any rather than State because the distributed realisation carries its
    # own state type; the schemes treat it as opaque and hand it back to the model.
    def step(self, model: Model, state: Any, dt: float) -> Trial: ...

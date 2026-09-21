"""The seam a discretisation supplies so the two schemes are written once.

`solver/schemes/` reach a model only through the members below, which is what lets
the same `SDIRK2` and `SDIRK2MRSAV` code drive the serial MAC model
(`solver/mac_ns.py`) and the distributed one (`solver/mac_parallel/`). Confirmed
scope, migration slices and test seams: `docs/model-seam-spec.md`.

Two deliberate typing choices:

- Payloads that differ between realisations (the packed velocity, the pressure field,
  the state, the trial and the stage) are typed `Any`. The serial model uses an
  ndarray plus `core.State`; the distributed model uses PETSc vectors plus its own
  state and stage types. The schemes treat them as opaque values passed back to the
  model, so the seam stays honest instead of pretending one type serves both.
- `force` is declared as a data member rather than a method, because the models store
  it as a callable attribute (the cavity load and the manufactured forcing both
  replace it at run time).
"""
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

class SolveResult(Protocol):
    """What a shifted Stokes solve returns: a velocity, a pressure and its own checks."""
    velocity: Any
    pressure: Any
    residual: float
    divergence_inf: float

class Model(Protocol):
    """Everything the schemes and the drivers require of a discretisation."""

    nu: float
    force: Callable[[float], Any]

    def vector(self, state: Any) -> Any:
        """Pack a state into the model's velocity representation."""

    def state(self, t: float, velocity: Any, r: float = 0.) -> Any:
        """Build a state at time `t` from a velocity and the scalar r."""

    def nonlinear(self, velocity: Any) -> Any:
        """Convective term N(v) on the model's velocity representation."""

    def nonlinear_with_lifting(self, velocity: Any) -> tuple[Any, float]:
        """Return convection and its weak pairing with the fixed boundary lifting.

        Zero scalar for models without an explicit lifting. SAV combines this scalar
        with the same stage weights as convection, then adds it to <B,stage velocity>.
        """

    def combine(self, *terms: tuple[float, Any]) -> Any:
        """Sum of scalar*field pairs, accumulated in the order given.

        Works for any field the model carries (packed velocity or pressure), so a
        stage can build its right-hand side without knowing the vector type.
        """

    def apply_K(self, velocity: Any) -> Any:
        """Discrete negative Laplacian acting on the velocity."""

    def apply_G(self, pressure: Any) -> Any:
        """Discrete gradient: pressure to velocity."""

    def apply_D(self, velocity: Any) -> Any:
        """Discrete divergence: velocity to pressure. G = -D^T for this pair."""

    def inner(self, left: Any, right: Any) -> float:
        """Velocity-space inner product, including the cell-area weight."""

    def max_abs(self, field: Any) -> float:
        """Infinity norm, used by the stage residual and divergence checks."""

    def solve(self, rhs: Any, *, mass: float, viscosity: float) -> SolveResult:
        """Shifted Stokes solve: (mass*I + viscosity*K) v + G p = rhs, D v = 0."""

    def solve_columns(self, columns: Sequence[Any], *, mass: float, viscosity: float) -> list[SolveResult]:
        """One solve per right-hand side, in the order given.

        The SAV stage needs two right-hand sides sharing a matrix. A backend that can
        batch them may do so internally and split the columns, which is bit-exact;
        one that cannot may loop.
        """

    def stage(self, pressure: Any, residual: float, divergence_inf: float, r: float = 0.,
              candidates: Sequence[float] = (), root_residuals: Sequence[float] = (),
              scalar_residual: float = 0.) -> Any:
        """Build a stage record, taking ownership of any payload the model allocated."""

    def diagnostics(self, state: Any) -> dict[str, float]:
        """Kinetic energy, modified energy, divergence, H1 seminorm and r."""


if TYPE_CHECKING:
    # Both realisations are checked against the seam here rather than in a test: the type
    # checker only inspects the packages in mypy's `files`, which exclude tests/, so a
    # conformance assignment inside a test file would never be read. A missing or wrongly
    # typed member fails these two assignments.
    from .mac_ns import MACNavierStokes
    from .mac_parallel.integrate import ParallelNS
    _serial_realisation: type[Model] = MACNavierStokes
    _distributed_realisation: type[Model] = ParallelNS

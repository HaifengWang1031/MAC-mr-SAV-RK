# FEM run / load / reconstruct / analyze

Independent deterministic rectangular Taylor–Hood P2/P1 workflow. Reuses the six
registered time schemes and common fixed/prescribed-step integrator; no adaptive
mode, new numerical formulas, legacy record migrations or campaign framework.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.fem_ns.run \
  --config experiments/fem_ns/configs/smoke.json --root /tmp/fem-smoke
# Pass the printed run directory explicitly; analysis never starts computation.
uv run python -m experiments.fem_ns.analyze /tmp/fem-smoke/runs/fem_ns/<run-id> \
  --root /tmp/fem-smoke
```

```python
from pathlib import Path
import numpy as np
from experiments.fem_ns.run import run, load_fem
from experiments.fem_ns.reconstruct import reconstruct
from experiments.fem_ns.analyze import analyze

root = Path('/tmp/fem-smoke')
directory = run({'nx': 8, 'ny': 4, 'T': .02, 'steps': [.007, .013],
                 'scheme': 'sdirk3_mrsav'}, root=root)
config, manifest = load_fem(directory, require_complete=True)
fields = reconstruct(directory, np.array([[.3, 1.1], [.2, .7]]))
initial_snapshot = reconstruct(directory, np.array([[.3], [.2]]), snapshot=0)
report = analyze(directory, root=root)
```

Use exactly one of `dt` or `steps`; omitted time settings use T=.02, dt=.01.
Defaults: nx=8, ny=4, rectangle 2x1, nu=.2, amplitude=.1, quadrature_order=8,
scheme=sdirk2_mrsav, gamma=1, r0=0, cache_size=4, stokes_tolerance=1e-9.
Other scheme names are those in `solver.schemes.registry.SCHEMES`. `snapshots`
defaults to `[0,T]`; `[]` retains none. Requested snapshots use nearest scheduled
nodes, with earlier-node ties; actual accepted times are authoritative. Only
accepted snapshots survive a failure. Unknown keys and invalid parameters fail
before creating a run.

## Smooth reference problem

The single case is a continuous manufactured Navier–Stokes solution with
`psi = amplitude exp(-t) x²(lx-x)² y²(ly-y)²`, velocity `(psi_y,-psi_x)`,
physical pressure `amplitude exp(-t)(x+2y-lx/2-ly)`, and analytic body force
`u_t + (u.grad)u - nu laplacian(u) + grad(p)`. Initial velocity uses consistent
L2 projection followed by a mass-only weak-incompressibility solve. This is not
a discretely exact solution; the tiny smoke configuration is not a spatial or
temporal convergence study.

## Saved representation and trust boundary

Runs live in `runs/fem_ns/<run-id>`, reports in `reports/fem_ns/<run-id>`.
Existing config/manifest/HDF5/log foundation and atomic pending-file publication
are reused. Identity includes effective config, source hash, Python/platform,
installed dependencies including scikit-fem, and thread environment. Reuse
requires complete compatible records and verified config/result checksums.
`rerun=True` always creates a new attempt without changing existing files.

HDF5 retains full scalar P2 component coefficients (including edge DOFs and wall
zeros), all final accepted stage P1 pressure coefficients, mesh coordinates and
connectivity, element identity/orders, library version, component/free/boundary
maps, basis DOF locations and element maps. Metadata explicitly distinguishes
FEM from MAC. Reader checks these against the deterministic declared space,
coefficient shapes/finiteness/constraints, pressure integral gauge, accepted
schedule/time consistency, diagnostics, snapshot mapping, stages and root
records. Missing/incompatible metadata is rejected even if a checksum was
refreshed. Reading requires the saved scikit-fem version to match the installed
version; old code hashes alone do not prohibit readonly analysis. Checksums
provide integrity, not authenticity against deliberate record rewriting.

`reconstruct` evaluates the saved mesh/bases without a dynamical model. It returns
`velocity` (2,n), `stage_pressure` (stages,n) for the final accepted step, `time`,
`r`, full `u_coefficients`/`v_coefficients`, and `pressure_semantics`. Pressure is
None before any accepted step and for snapshot requests: pressure is **not**
stored for every snapshot. Snapshot index addresses the saved list, not an
arbitrary trajectory node. Points must be finite (2,n) coordinates in the domain.

Analysis opens inputs readonly, constructs no Model/backend, and writes only
`analysis.json`, `history.csv`, `fields.csv` and `analysis.log` in reports. The
field CSV contains evaluated velocity and final-stage pressures on a small
17x13 display grid, not a replacement coefficient representation. Analysis root
inside the input run is rejected. It labels failed prefixes explicitly.

## Validation and limits

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest tests/test_fem_workflow.py -q
uv run mypy experiments/fem_ns
```

Tests cover actual six-scheme fixed/given integration roundtrips, evaluated
velocity/pressure agreement, a known quadratic edge basis and analytic P1 field,
readonly input-byte invariance with dynamical construction guards, reuse/rerun,
provenance changes, refreshed-checksum metadata corruption, zero/one accepted
step failures and compensated long-schedule midpoint ties.

Physical divergence diagnostics are finite-element quadrature-point maxima and
L2 norms, distinct from full weak continuity residuals. Stage pressures keep
registry multiplier semantics; no same-order physical endpoint-pressure claim.
No PDE convergence or performance conclusion follows from workflow tests.
Failed steps save accepted prefixes; setup/storage failures retain a failed
manifest/log and re-raise, and records without valid results remain unreadable
and unreusable. Atomic rename is not an fsync/durable transaction guarantee;
interrupted pending/running records are not reused. No checkpoint continuation.
Full-project mypy has six known pre-existing errors outside FEM; owned modules
are checked separately. Integration validation documentation is maintained by
the validation owner, using the external ticket03 evidence handoff.

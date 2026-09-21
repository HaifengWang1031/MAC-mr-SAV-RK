# Confirmed model-seam specification

Status: confirmed 2026-09-21 — the test seams and the rounding gate below were
confirmed by the user, so implementation may proceed slice by slice. Source of the
request: the plan to add finite-element and spectral solvers.

## Purpose

`solver/schemes/` runs only on the serial MAC model (`solver/mac_ns.py`), while
`solver/mac_parallel/integrate.py` re-implements the same two stage schemes against
PETSc vectors (130 lines against 73). Every further discretisation multiplies that
duplication: discretisations × backends × schemes. This specification extracts the
interface a discretisation must supply so that `SDIRK2` and `SDIRK2MRSAV` are written
once and used by every discretisation, existing and future.

## Confirmed scope

- Type/interface refactor only. No numerical formula, tolerance, gate or
  discretisation changes; any behavioural difference is a defect, not a refinement.
- `solver/mac/` (SciPy/SuperLU serial MAC) keeps its current API. `State` keeps named
  `u, v` arrays, as CONTEXT.md confirms, so the serial experiments and every existing
  record stay valid.
- `solver/mac_parallel/` keeps its distributed assembly, block preconditioner, option
  plumbing and divergence-aware stopping. Only its duplicated stage equations go.
- `fem/` and `spectral/` are out of scope here; they are the reason the seam exists.
- Out of scope: two-dimensional domain decomposition, parallel HDF5, multi-node runs,
  new schemes, adaptive control.

## The seam

The schemes already reach the model only through `model.vector(state)`,
`model.state(t, velocity, r)`, `model.force(t)`, `model.nonlinear(v)`,
`model.backend.solve(rhs, mass=…, viscosity=…)`, `model.nu`, `model.ops.K/G/D`,
`model.grid.inner/norm` and raw NumPy arithmetic on the packed velocity. The part
that cannot be shared today is the arithmetic and the operator access, so the seam
adds exactly those and names the two existing variants consistently:

| member | serial MAC supplies | distributed MAC supplies |
|---|---|---|
| `vector(state) -> V` | pack `u, v` | `state.velocity` |
| `state(t, V, r) -> State` | unpack | `DistributedState` |
| `zero_like(V)`, `duplicate(V)` | `np.zeros_like` | `Vec.duplicate` |
| `combine(terms)` | sum of scalar×array | the existing `combine` |
| `apply_K(V)`, `apply_G(P)`, `apply_D(V)` | sparse matvec | `Mat.mult` |
| `inner(a, b)`, `max_abs(V)` | `grid.inner`, `np.max` | `Vec.dot`, `Vec.norm(INF)` |
| `nonlinear(V)` | Numba convection | halo convection |
| `solve(rhs, mass, viscosity)` | `DirectStokes` | `ParallelStokes` |
| `stage(...) -> Stage` | NumPy `Stage` | records then destroys PETSc vectors |
| `diagnostics(state)`, `force(t)` | current | current |

`Stage.pressure` stays opaque to the schemes, with ownership and lifetime held by the
model through `stage(...)`, because the serial path stores an array and the
distributed path an owning PETSc vector.

## Migration in confirmable slices

1. Introduce the protocol and make the serial model satisfy every member through
   behaviour-identical wrappers (`combine` over NumPy, `apply_K` = `ops.K @ V`, …).
   Gate: the entire existing suite passes unchanged.
2. Rewrite `sdirk2.py` and `sdirk2_mrsav.py` against the protocol only.
   Gate: the scheme, integration and workflow tests pass unchanged, and the deviation
   from pre-refactor output is measured and recorded below. Seam 1 belongs to slice 3,
   not here: it needs *both* models to satisfy the seam, and the first wording of this
   slice implied otherwise — a sequencing error in this specification.
   Landed 2026-09-21; see "Slice 2 rounding measurement".
3. Drive the shared schemes through `PETScStokes` (single-rank PETSc adapter).
   Gate: agreement with the SuperLU reference.
4. Drive them through `ParallelStokes` on 2 and 4 ranks.
   Gate: agreement with the serial reference, as `tests/test_parallel.py` does now.
5. Delete the duplicated stage code and its driver hooks from
   `mac_parallel/integrate.py`, keeping the model, layout, halo exchange and stopping
   logic.

Each slice is independently revertible and leaves the repository green.

## Slice 2 rounding measurement

Measured 2026-09-21 by replaying one deterministic case through the pre-refactor code
(which is slice 1: it changes no scheme arithmetic, so its output is the original
one) and then through the rewritten schemes: 16×12 grid, `nu=0.1`, `amplitude=0.2`,
`dt=1e-3`, five steps, both schemes, comparing every stage pressure, every stage
scalar, the final velocity, the final `r` and the diagnostics — 46 arrays. The harness
uses only the public `scheme.step` API, so it runs against either revision.

| observation | value |
|---|---|
| arrays bitwise identical | 14 of 46 |
| SDIRK2 worst relative deviation | 4.434e-12, on an incremental stage pressure at step 2 |
| SDIRK2-mr-ccSAV worst relative deviation | 2.833e-16, i.e. roundoff |
| SDIRK2 final velocity | 1.232e-14 absolute on a field of scale 0.726 (≈1.7e-14 relative) |
| SDIRK2-mr-ccSAV final velocity | bitwise identical |
| final `r`, both schemes | bitwise identical |

Interpretation: the shift is roundoff, not a formula change. SDIRK2's largest
relative deviation sits in the *incremental* stage pressure, which is the
velocity-level roundoff amplified by that record's `1/dt` factor (1e-14 × 1e3); the
quantity is already documented as an incremental multiplier rather than a physical
pressure. SDIRK2-mr-ccSAV stays at the ulp level because its velocity and `r` updates
happen to keep the same accumulation order. No convergence rate, gate or tolerance
changed.

## Slice 3 record (2026-09-21)

`ParallelNS` satisfies the seam. The conformance is verified by the type checker rather
than asserted: `solver/model.py` assigns both realisations to `type[Model]` under
`TYPE_CHECKING`, and renaming one member makes mypy fail at that line, so the check is
live. (The original version put that assignment in a test, which never worked — mypy
only inspects the packages in its `files` setting, and `tests/` is not one of them.)

Seam 1 is `tests/test_parallel.py::test_one_scheme_object_drives_both_models`: one
scheme object per case drives the serial model and the single-rank distributed model
(`COMM_SELF`, so it behaves identically however pytest is launched) and requires
agreement below 1e-8 on the packed velocity, the scalar `r` and the stage scalars, for
both schemes. It lives in the parallel test file rather than beside the member seams
because it needs petsc4py, and that is the file that skips cleanly without it.

Two decisions the member table did not anticipate:

- `apply_K`, `apply_G` and `apply_D` allocate PETSc vectors that the schemes cannot
destroy, so the model owns one reused buffer per operator and releases them in
`close()`. The `apply_D` buffer must take the *full* pressure layout: the solver's
system drops one continuity row but the seam's `apply_D` returns all rows, matching the
serial operator. The first version used the reduced layout and seam 1 caught it
(`MatMult ... Nonconforming object sizes: global dim 108 107`).
- `apply_K` is deliberately not an alias of the existing `viscous`, because
  `ParallelSDIRK2` destroys the vector it receives; keeping them separate avoids
  handing a destroyed buffer back until slice 5 removes that stepper.

Gate: 7 parallel tests pass on 1, 2 and 4 ranks in a dedicated environment
(`docs/parallel.md` records the verified build recipe and the corrected diagnosis of the
MPI mismatch), the serial suite is unchanged, and mypy is clean.

## Slice 4 record (2026-09-21)

`test_distributed_time_steps_match_serial_sdirk` now drives the *shared* scheme — one
instance per case — through the distributed model on 2 and 4 ranks and through the
serial model on rank 0, over three step sizes, comparing the packed velocity, the
scalar `r` and every stage root. That is seam 1 at scale. It also checks that every rank
selects the same roots, which is what the removed rank-0 root broadcast used to
guarantee: the shared scheme derives the cubic from globally reduced inner products, so
each rank computes identical coefficients.

A hypothesised ownership problem was refuted by measurement instead of designed around.
The shared schemes discard the intermediate vectors a solve returns (the raw velocities
and pressures before `combine` builds the stage values), and the removed `ParallelSDIRK2`
destroyed those explicitly, so a leak looked likely — at 1024² it would have been roughly
eight vectors per step, about 64 MB. Measured on a 128² grid over 25 steps, peak RSS grew
by 0.6 MB for SDIRK2-mr-ccSAV and 0.0 MB for SDIRK2, against the ~30 MB the leak would
have produced at that size. No leak exists: CPython's reference counting collects the
petsc4py wrappers promptly and they destroy their PETSc objects. The seam therefore gains
no `release` member, and the only residual condition is that this rests on the wrappers
becoming unreachable — if a later change keeps a reference to an intermediate, its
lifetime becomes that reference's, which is ordinary Python behaviour rather than a seam
gap.

Gate: 7 parallel tests pass on 1, 2 and 4 ranks; the serial suite is unchanged.

## Slice 5 record (2026-09-21)

The duplicated stage layer is gone. `ParallelSDIRK2` — 52 lines re-implementing both
schemes against PETSc vectors — and `DistributedTrial` are deleted from
`solver/mac_parallel/integrate.py`, which drops from 191 to 131 lines. Its imports of
`ETA`, `DELTA` and `real_roots` go with it, so the stage formulas now exist only in
`solver/schemes/`: that duplication was the reason this refactor exists. What remains is
the discretisation and its distribution: `SlabLayout` with the halo exchange,
`DistributedState`, `DistributedStage`, `ParallelNS` with the seam members, `combine`,
and the block-preconditioner and stopping logic in `stokes.py`.

`experiments/mac_parallel/run.py` builds a shared scheme instead of the removed stepper.
Its distributed payloads are typed `Any`, with a comment saying why: `core.Trial` carries
the serial `State` and `Stage` types, so a driver that holds PETSc vectors cannot use
them without a lie. That is the price of keeping `State` as CONTEXT.md confirms it, and it
is confined to the two parallel drivers.

Gate: the parallel tests pass on 1, 2 and 4 ranks; a real CLI run under `mpiexec -n 2`
and `-n 4` finishes complete, writes its record and is reused on a second invocation (5
steps and 10 steps respectively); the serial suite is unchanged at 27 passed with
`test_parallel.py` skipped; mypy is clean over 43 files.

The refactor is not a line-count win — `solver/model.py` and the seam members add more
than the 60 deleted lines — it removes the duplication that a third discretisation would
have multiplied. The payoff is the next discretisation, not this diff.

## Risks

- The schemes are validated numerical code and the seam touches their arithmetic.
  Mitigation: the slicing above, plus independent references (SuperLU, the serial MAC
  path, the ghost-point stencil) instead of assertions recomputed the way the code
  computes them.
- Rounding: the confirmed decision is that the serial `combine` stays generic (a
  term-by-term `axpy` accumulation), so its association order differs from today's
  hand-written expressions and results shift at roundoff. Measured consequence and
  gate:

  * Slice 1 changes no scheme arithmetic — it only adds wrappers — so its output is
    unchanged by construction, and the existing suite passing unchanged is the whole
    gate.
  * Slice 2 introduces the shift. Its gate is agreement with pre-refactor output at a
    low relative tolerance whose measured value is reported in the section below
    rather than declared in advance. Identity hashing already treats a code change as
    a new run identity, so nothing is silently reused; the point of measuring is to
    show the shift is roundoff and not a formula change.
  * `docs/validation.md` gained that note when slice 2 landed, because the tables
    recorded there were produced by the pre-refactor expression order and are no
    longer bitwise comparable.
- The distributed SAV stage currently broadcasts the selected root from rank 0. With
  a global-reducing `inner`, every rank derives identical coefficients, so the
  broadcast becomes redundant; removing it belongs to slice 4 and must not change the
  selected root, which is itself residual-checked.

## Confirmed test seams

Confirmed 2026-09-21. No test is written outside these seams.

1. **Scheme versus model** (new, confirmed): one test drives *the same* scheme object
   through the serial MAC model and through the single-rank distributed model, for
   both SDIRK2 and SDIRK2-mr-ccSAV, and requires agreement. This is the invariant the
   refactor creates: today only two separate implementations are compared, afterwards
   one implementation must serve both models, so an incomplete seam fails here
   directly.
2. **Distributed versus serial** (existing, unchanged):
   `tests/test_parallel.py::test_distributed_time_steps_match_serial_sdirk` for both
   `sav=False` and `sav=True`, as the gate for every slice.
3. **Seam members** (new, confirmed in full): all four adapted members — `combine`,
   `apply_K`, `inner`, `max_abs` — are checked against independent references: the
   pre-refactor serial expressions for the first two, and NumPy applied to gathered
   fields for the last two. A wrong wrapper must fail here rather than drift into a
   scheme-level difference.
4. **Unchanged seams** (confirmed): the agreed list in `docs/spec.md` (public MAC
   operators and boundary behaviour, JIT kernels versus an independent reference,
   Stokes/stage residuals and root behaviour, integration/convergence, run/save/load/
   analyze/reuse/rerun) continues to gate every slice.

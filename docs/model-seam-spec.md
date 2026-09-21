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

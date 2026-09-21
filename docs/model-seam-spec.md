# Confirmed model-seam specification

Status: proposed 2026-09-21. Implement only after the test seams in the last section
are confirmed. Source of the request: the plan to add finite-element and spectral
solvers.

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
   Gate: the scheme, integration and workflow tests pass unchanged.
3. Drive the shared schemes through `PETScStokes` (single-rank PETSc adapter).
   Gate: agreement with the SuperLU reference.
4. Drive them through `ParallelStokes` on 2 and 4 ranks.
   Gate: agreement with the serial reference, as `tests/test_parallel.py` does now.
5. Delete the duplicated stage code and its driver hooks from
   `mac_parallel/integrate.py`, keeping the model, layout, halo exchange and stopping
   logic.

Each slice is independently revertible and leaves the repository green.

## Risks

- The schemes are validated numerical code and the seam touches their arithmetic.
  Mitigation: the slicing above, plus independent references (SuperLU, the serial MAC
  path, the ghost-point stencil) instead of assertions recomputed the way the code
  computes them.
- The serial `combine` must reproduce the current expression order and temporaries;
  otherwise every recorded result shifts at roundoff, identities change and stored
  comparisons stop matching. Slice 1 must show bitwise-identical scheme output.
- The distributed SAV stage currently broadcasts the selected root from rank 0. With
  a global-reducing `inner`, every rank derives identical coefficients, so the
  broadcast becomes redundant; removing it belongs to slice 4 and must not change the
  selected root, which is itself residual-checked.

## Proposed test seams (needs confirmation before any test is written)

1. **Scheme-versus-model** (new): one test drives both schemes through the serial MAC
   model and through the single-rank PETSc model and requires identical results,
   extending today's adapter test from one SDIRK2 step to both schemes.
2. **Distributed-versus-serial** (existing, must keep passing unchanged):
   `tests/test_parallel.py::test_distributed_time_steps_match_serial_sdirk` for both
   `sav=False` and `sav=True`.
3. **Seam members** (new): `combine`, `apply_K`, `inner` and `max_abs` are checked
   against independent references — the current serial expressions, and NumPy applied
   to gathered fields — so a wrong wrapper cannot pass by construction.
4. **Unchanged seams**: the agreed list in `docs/spec.md` (public MAC operators and
   boundary behaviour, JIT kernels versus an independent reference, Stokes/stage
   residuals and root behaviour, integration/convergence, run/save/load/analyze/
   reuse/rerun) continues to gate every slice.

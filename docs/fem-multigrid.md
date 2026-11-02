# Shared FEM solver comparison and optimization

Status: implemented, measured and release-checked on 2026-11-02. Existing solver
and experiment defaults are unchanged. The selected opt-in path is weak momentum
with recycled LU; explicit consistent-mass CG/METIS choices cover large models.
GAMG remains experimental and was not selected as a default.

Results: [versioned summary](fem-scalability-results.md) and
[qualified presets](fem-solver-presets.json). The complete per-sample report remains in
`reports/fem_multigrid/20261102T083710-final/report.md` as a local generated artifact.
Raw fixtures, immutable original/candidate sources, workers and logs are in
`runs/fem_multigrid/20261102T083710/`. That pre-edit baseline includes the user's
existing uncommitted work; no checkout/reset was used.

## Public controls and measured selection

All four families (closed FEM/cavity, cylinder, wedge, Hele–Shaw) accept explicit
optional settings. New keys are not inserted into old default configurations.

```json
{"stokes_backend": "recycled", "fem_load_mode": "weak"}
```

This removes repeated mass-inverse round trips. On the medium saved meshes,
12-step SDIRK3-mr-ccSAV prescribed-step replay reduced online time by 3.5--9.7%
relative to the frozen original recycled solver (three rotated repeats). Fixed-step
short runs improved by 6.0--10.8%; PI adaptivity was problem-dependent: the short
cavity group was 1.2% slower, the other three groups improved 2.7--6.5%. These
numbers do not establish a universal default or long-time production speedup.

For a large consistent mass system, independent choices are:

```json
{"stokes_backend": "recycled", "fem_load_mode": "weak", "mass_backend": "metis"}
```

or `mass_backend="cg"`. METIS changes only the scalar mass-factor ordering;
CG applies the same consistent mass matrix with positive diagonal preconditioning,
rtol=1e-13 and no lumping. Identical uncoupled components still share a scalar
factor/operator. At 730344 velocity DOFs, original MMD mass-LU initialization
exceeded a 90s limit; CG initialized around 5--7s and METIS mass LU around 8s.
Large CG and METIS timings are separate batches, not an interleaved global ranking.
The largest experiment measures one cold complete SAV step, not a production
long-run break-even point. Detailed initialization, online, pressure-recovery,
end-to-end and peak-RSS numbers are retained in the report.

## Shared momentum representation

`solver/momentum.py` preserves the original Riesz/coefficient arithmetic by default
for all models. Explicit FEM weak mode uses `solver/fem/momentum.py`: momentum
loads remain `M*u`, `K*u`, `-B.T*p` and raw convection loads. Physical state vectors,
RK coefficients, pressure scaling, SAV polynomials and root selection are shared
across all six time formats. Fixed-trace contributions affect only the first
Stokes response column; SAV correction columns remain homogeneous. Volume and
boundary quadratures and lifting work are unchanged.

The actual post-root coefficient defect still receives a full consistent mass
solve. Only an extra inverse needed for denominator normalization is avoided:

\[
 c_M=\|D_M^{-1}M\|_\infty,\qquad
 \|D_M^{-1}F\|_\infty/c_M\le\|M^{-1}F\|_\infty.
\]

The lower-bound denominator makes acceptance stricter than the original
coefficient-relative check. A diagonal residual never replaces its numerator.
Original weak momentum, coefficient residual, every continuity row and pressure
gauge remain required. Candidate CG accuracy is also checked against the exact
METIS-mass reference where original large MMD construction does not finish.

## Optional PETSc backend

Explicit `stokes_backend="petsc"` uses the exact mixed outer matrix and FGMRES.
For H=m*M+s*(A+kappa*M), the pressure auxiliary inverse is

\[
 \widehat S^{-1}=(m+s\kappa)\widehat K_D^{-1}+sD_p^{-1},
 \qquad K_D=B\operatorname{diag}(M)^{-1}B^T.
\]

Positive P1 pressure weights define D_p. Only the auxiliary closed-domain pressure
operator is anchored; the original outer system retains every B row and its
weighted mean gauge. Open traction pressure has no artificial mean subtraction.
Two cycles approximate the pressure auxiliary inverse. Identical velocity blocks
share a hierarchy; pure mass uses Jacobi, ordinary shifted systems one velocity
cycle, pure stiffness two cycles. Nearby positive shift coefficients can reuse a
hierarchy within a factor two; the outer operator always uses the exact new shift.

If Krylov convergence fails the original checks, up to two solves correct the
true mixed-system defect, checking all equations again. Any eventual global LU
fallback is counted. Such a record is disqualified as an AMG-only speedup.
The initial and refined candidates are separately frozen: the refined candidate
passed all 24 matrix samples without global LU, including five shift regimes
and two nonzero RHS per case. It did not win on speed. Two current cylinder
full-step/adaptive samples also exceeded the stricter 1e-7 stage-pressure agreement
gate (about 1.68e-7 and 2.96e-7), while independent endpoint physical-pressure
errors were about 1.54e-10 and 2.83e-10. These samples remain disqualified; no
acceptance threshold was relaxed. This distinction is why AMG is not a selected
production preset in this comparison.

The adapter uses COMM_SELF and real float64. Distributed FEM assembly, MPI scaling
and GPU acceleration are not implemented here. GAMG was measured; the local PETSc
has no Hypre. Selecting `stokes_amg_type="hypre"` requires a compatible native
build and has an explicit availability error, not a silent substitution.

## Environment and verification

The original `.venv` and running frozen-source Re=10000 job were not modified.
An isolated `.venv-fem-mg` uses PETSc/petsc4py 3.25.5 and matching Open MPI.
A cached MPICH wheel failed an ABI import probe; rebuilding against the local
Open MPI PETSc resolved it. On this Mac the binding was built with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-fem-mg uv sync --locked --extra mesh --group dev
uv pip install --python .venv-fem-mg/bin/python Cython setuptools
PETSC_DIR=/opt/homebrew/opt/petsc PETSC_ARCH='' \
MPICC=/opt/homebrew/opt/open-mpi/bin/mpicc CC=/opt/homebrew/opt/open-mpi/bin/mpicc \
uv pip install --python .venv-fem-mg/bin/python --no-cache --no-binary petsc4py \
  --no-build-isolation --no-deps petsc4py==3.25.5
```

The optional `multigrid` extra and `threadpoolctl` dev dependency are locked.
The complete isolated test environment also includes `--group notebook`; omitting
it skips notebook coverage. Provenance adds PETSc build details only for explicit
PETSc computations; saved-record loading and wedge analysis do not require PETSc.

Release results: ordinary environment **621 passed, 21 skipped**; PETSc environment
**632 passed, 10 skipped**; affected **71 sources pass mypy**. The ordinary environment
was kept unchanged: a temporary PYTHONPATH provided only its missing newly declared
threadpoolctl development dependency. Full-project mypy still reports six existing
errors in three untouched Kolmogorov scripts. Logs are in the run's `validation/`.

Focused tests cover independent SPD mass solutions, exact manufactured mixed
solutions, immutable batched RHS, full affine-trace/reaction SAV stages, all six
weak-load formats, exact defect refinement, and all four public run/load workflows.

## Replaying and reading the comparison

`experiments/fem_scalability` keeps computation and analysis separate. Preparation
requires an explicit pre-edit `baseline_source` in the run directory; the controller
then freezes the candidate and saves actual meshes, common initial states and mass
matrices. Fresh-process numerical threads are limited to one, variant order rotates
between repeats, and accepted-state diagnostics remain enabled. Five two-RHS shift
regimes, fixed steps, prescribed changing steps and native PI are distinct workloads.
Larger meshes additionally measure one cold full four-stage SAV step.

Reports never construct a flow model or run time stepping:

```bash
.venv-fem-mg/bin/python -m experiments.fem_scalability.report \
  --run runs/fem_multigrid/20261102T083710 \
  --output reports/fem_multigrid/reanalysis
```

A repeated measurement request checks existing source/options and reuses immutable
samples. New measurements should use a new campaign or explicit additional repeats.
Raw failures are retained. An initial adaptive metadata writer rejected missing-history
NaNs after successful numerical integration; those records remain archived. The corrected
worker stores nonfinite trial metadata as null plus original index/key/value records,
without changing the controller or accepting nonfinite physical fields.

There are 296 formal samples in 104 groups; 282 pass all applicable comparison gates.
Failures, counted fallbacks and pressure disagreements are not averaged into qualified
speedups. Parent routing helpers evolved between phases; frozen numerical source and
worker hashes, actual arguments and controller file hashes at record write are retained.
The machine also carried the untouched prior Re=10000 job; these are shared-machine
single-process measurements, not dedicated-node scalability or long-time physics evidence.

Primary references: [PETSc block solvers](https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/),
[FGMRES](https://petsc.org/release/manualpages/KSP/KSPFGMRES/),
[Creff–Guermond generalized Stokes comparison](https://arxiv.org/html/2407.01783v2).

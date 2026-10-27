# Confirmed implementation specification
2026-10-27 confirmed FEM extension: independent scikit-fem continuous Taylor–Hood
P2 velocity/P1 pressure on deterministic rectangular triangular meshes, homogeneous
no-slip, consistent mass/Riesz coefficient representation and antisymmetric convection.
Unpenalized direct shifted Stokes checks every weak continuity row and physical-integral
pressure gauge; physical gradient divergence is reported separately. All six shared
formats support fixed/prescribed steps. Dedicated run/load/reconstruct/readonly analysis
retains full coefficients and mesh/DOF metadata; legacy layouts remain unchanged.
No adaptive or nonhomogeneous-boundary extension, production campaign or PDE time-order
proof. See [FEM conventions](fem.md) and [measured evidence](validation.md#2026-10-27-fem-release-verification).

2026-10-05 Legendre performance extension: cached immutable internal quadrature,
exact 1D closed forms, tensor mass solves, shared weak operators and batched RHS.
Default direct Stokes retains all equation/physical-divergence checks. Optional
spectral tensor Stokes uses reusable 1D generalized eigenspaces and true Schur-CG;
all six schemes and existing I/PI rejection/endpoint rules are exercised separately.
No spatial, convection, lifting or pressure semantics change. See docs/legendre-optimization.md.
2026-10-05 illustrative TS instability: `case=ts_wave` adds an experimental initial
condition, not new solver equations. It stores the linearised MAC eigenvalue/unit
seed; positive growth is qualified against an independent continuous reference,
short time/grid refinements and a stable control before nonlinear visualisation.
The TS movie uses an explicitly labelled moving coordinate frame. Existing cases,
operator/stage formulas and laboratory-frame saved fields retain their semantics.

2026-10-04 confirmed first-order extension: independent IMEXEuler/IMEXEulerMRSAV
with old-time explicit convection/forcing and constrained backward Euler viscosity.
SAV uses G=1-r, Q=1, two common-matrix RHS solves and one unique linear scalar solution.
One-stage Trial, complete residuals/continuity, fixed and prescribed steps, all four
Model implementations and public MAC/periodic/channel run/load/analyze paths.
Registry metadata replaces stage/order guessing in these entry points. Legacy
methods and default comparison groups retain their conventions. Euler has no
embedded estimate in this release. See docs/imex-euler.md.

2026-10-04 confirmed channel extension: independent `ChannelGrid/ChannelOperators/ChannelModel`
under `solver/channel`, periodic-x and stationary no-slip-y second-order MAC. Real full
face arrays preserve the existing shape convention but identify the duplicated periodic
u endpoint; packed velocities count only independent faces. Shared DirectStokes uses a
structural operator protocol with its existing algorithm unchanged. All four schemes,
fixed/prescribed/adaptive drivers, mean-flow forcing, complete residual checks and dedicated
run/load/analyze workflows are covered. The subsequent optional FFT/banded backend
solves the identical MAC discretisation and is compared to direct solves before timing.
The default remains direct. See docs/channel.md.

Source: user-confirmed design in the current task, 2026-09-20.

- Separate desktop project MAC-mr-SAV-RK; source Stokes_BC.py and Fourier project remain unchanged.
- Uniform rectangular MAC, homogeneous no-slip, mean-zero pressure, compatible divergence/gradient, positive viscous matrix, energy-compatible centered convection.
- Numba local kernels, no fastmath initially; direct sparse coupled Stokes backend and bounded factorization reuse; an interface for future iterative backends. Confirmed 2026-09-21: a discretisation-agnostic model seam so the two schemes are written once (`docs/model-seam-spec.md`); `solver/model.py` is that seam, both schemes reach the problem only through it, and `solver/spectral/` is a second discretisation behind it (`docs/spectral.md`), including the inhomogeneous moving-lid boundary as a `LidLifting` whose loads move to the right-hand side, so the saddle-point matrix is unchanged exactly as in the MAC lid load — which is what makes the seam's requirement of a single vector space observable, since the MAC model satisfies it silently through a diagonal mass matrix.
- Two schemes: ordinary IMEX-SDIRK2 and the notes' incremental SDIRK2-mr-ccSAV. Fixed and prescribed steps. Each SAV stage uses two common-matrix RHS solves and a cubic. Enumerate numerical real roots, validate residuals, choose minimum absolute r, save candidate roots/count/selection/residual. Handle degeneracy.
- State carries named u,v arrays and SAV r; separate incremental stage-pressure output.
- Experiments: lid-driven cavity (moving top lid, grid comparison, streamline/vorticity figures); the A/B steady-force rotation study on `(0,2*pi)^2` (net-torque versus zero-torque forcing, long-time boundedness, energy budget, angular-momentum reversals) with its own observables `angular_momentum`, `power` and `dissipation` in the model diagnostics; and the forced no-slip cavity (`f = (0, amplitude*sin(2*pi*m*x))` with an isotropic clamped-beam initial condition, fixed-step temporal accuracy of both schemes against a fine-step reference). The smooth manufactured fields of `experiments/problems.py` remain as the independent spatial/temporal reference inside the test suite rather than as runnable experiment kinds.
- Agreed test seams: public MAC operators and boundary behavior; JIT kernels versus independent reference; Stokes/stage residuals and root behavior; integration/run-save-load-analyze-reuse; initial-condition construction (beam values against a high-precision reference, wall compatibility, divergence and projection) and forcing sampling.
- runs/<experiment>/<run-id> has config.json, manifest.json, results.h5, run.log. Actual times, diagnostics, requested/actual snapshot mapping, final state and stage roots are retained. Analysis writes reports/<analysis>/<id> with analysis.json, figures, tables, analysis.log and explicit input run IDs.
- Identity includes effective config, code and environment. Reuse only verified complete compatible results. Preserve failures and accepted prefixes; rerun creates new identity attempt. No checkpoint continuation. Batches record config and members.
- Validate 32²,64²,128² and one non-square grid, algebraic structure, spatial/time convergence and workflows. Separate JIT warmup, factorization and integration timings.

Latest confirmed refinements: place modules directly under solver/ (no nextgen/). Manage the environment with uv, commit uv.lock, and use uv run for all commands.

2026-10-04 confirmed extension: an independent doubly periodic Fourier–Leray model under `solver/fourier/`, reusing all four existing SDIRK/mr-ccSAV methods and I/PI drivers. Real collocated exported fields, internal normalized complex rFFT coefficients, strict 2/3 filtering, unprojected nonlinear/force loads, full shifted velocity-pressure residuals, explicit mean-mode conventions. Dedicated `experiments/periodic_ns` run/load/analyze/validation workflow preserves existing MAC and Legendre APIs. Operator, stage, nonlinear MMS, steady infinite-modal spatial, adaptive and persistence seams are validated separately. See `docs/fourier.md`; no periodic MAC comparison baseline or long-time turbulence claim is included.

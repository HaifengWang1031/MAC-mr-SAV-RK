# Confirmed implementation specification

## Current capability summary

The implemented scheme registry contains six names: `imex_euler/imex_euler_mrsav`,
`sdirk2/sdirk2_mrsav`, and `sdirk3/sdirk3_mrsav`. They have velocity design orders
1/2/3 and 1/2/4 stages. All support fixed and prescribed positive steps; the four
SDIRK schemes additionally provide embedded I/PI adaptivity. Euler has no embedded
or step-doubling estimator. The five current spatial Models are no-slip MAC,
Legendre, doubly periodic Fourier, Channel MAC, and Taylor–Hood P2/P1 FEM.
Experiment-specific choices can be narrower than the common solver API.

See the [current capability table](../README.md#当前算法与支持范围),
[architecture](architecture.md), [registry](../solver/schemes/registry.py), and
[adaptive rules](../solver/adaptivity/README.md). The dated decisions below retain
their original scope; later extensions supersede initial exclusions. Their test
or production evidence applies only to the stated configurations and dates.

## Confirmed extensions and historical scope

2026-11-01 approved Hele-Shaw airfoil extension: separate pressure-driven MWR
inertial Brinkman model, fixed full-gap NACA0012, explicit closure coefficients,
optional zero-default consistent-mass reaction operator and independent scalar
Darcy reference. Reuse all six stages/root algorithms and existing adaptive drivers.
Pressure-traction ends differ from Darcy pressure Dirichlet data; no mean gauge.
Save actual meshes/maps, physical snapshot pressure, force semantics and all failure
prefixes. Readonly analysis never generates geometry or advances a flow.
No 3D applicability, unconditional open-boundary stability or grid-independent claim.
See [model and validation scope](hele-shaw.md).

2026-11-01 wedge-cavity implementation repair: new runs use r0=0, stationary
shared lid/wall end vertices, and the fixed-lifting skew pairing b(U,U,g) from
the same quadrature as N(w). Zero normal boundary flux does not eliminate this
volume pairing. Preserve old recorded r=1/corner conventions when loading legacy
runs. Fixed-step records and the final snapshot strictly reach T. The new saved-run
movie path reuses static artists and streams frames to ffmpeg; it never advances
the flow and does not present stage multipliers as endpoint physical pressures.
These changes do not establish steady-state or spatial/time-convergence qualification.

2026-10-30 FEM performance goal: retain the exact consistent-mass shifted system,
all continuity/gauge/coefficient checks and SAV/boundary semantics. The recycled
path first tries checked multicolumn defect correction, falls back to the existing
GMRES path, then a fresh robust factorization. Diagonal scaling only screens
when complete checks may succeed. Linear-response physical divergence contracts
the required derivatives directly at all quadrature points for multiple RHS; a single
response retains the full field cache for its next stage evaluation. Existing config keys
and saved-run identities are unchanged. Frozen-source replay, native adaptivity,
parallel factor engines, larger linear-system probes and size/format measurements
have qualified the selected strategies;
no global optimum or unmeasured acceleration is claimed. See [work and evidence](fem-optimization.md).

The same system now supports cached structural METIS ordering (`auto` at 20000
unknowns, or explicit `mmd`/`metis`), exact sharing of uncoupled identical mass
component factors, and optional pinned Linux x86_64 PARDISO with a local thread
limit. PARDISO's own ordering requires `auto`; it keeps two internal refinements.
Every path restores original DOF coordinates before all original checks and
retains robust SuperLU fallback. New options are optional canonical config keys;
derived adaptive cylinder campaigns preserve the explicit outlet and factor options.

2026-10-28 optional backflow-damped outlet: add half the negative normal speed
times velocity on natural facets and the identical boundary-work correction in
SAV. Natural is still the default. New case/boundary/pressure metadata and
nonnegative dissipation diagnostics separate this changed boundary model from
the original DFG reference. See [definition and limits](backflow-outlet.md).

2026-10-28 explicit forced minimum-step option: `integrate_adaptive` defaults to
`min_step_policy=fail`. The user-requested `force` option clamps ordinary step
proposals to the floor and accepts finite over-tolerance trials at exactly that
floor, retaining the original errors/acceptance records and complete stage checks.
Nonfinite states/errors or failed stages/diagnostics are not accepted; short final
alignment steps keep the existing error check. Cylinder configuration stores this
option only when explicit, preserving old canonical configuration identities.
Forcing does not establish that the requested local tolerances or physical
accuracy have been achieved.

2026-10-28 cylinder/adaptive FEM extension: existing six fixed formats and four
embedded SDIRK2/3 I/PI formats are reused without changing stages or SAV root
selection. DFG2D-2 Re100 uses a fixed parabolic inlet, no-slip cylinder/walls and
gradient-form natural outlet; verified polygonal P2/P1 meshes retain all actual
saved topology. Open pressure is ungauged, and physical force/probe pressure is
separate from incremental stage multipliers. Boundary-consistent SAV work is in
[ADR0001](adr/0001-open-flow-sav-boundary-work.md).

FEM optimization preserves consistent mass, all pressure-test rows and original
residual tolerances: cached P2 quadrature, one-call arbitrary force projection,
known MMS-mode preprojection, checked direct ordering and bounded exact-matrix
recycled-factor preconditioning. Old fixed records remain readable; adaptive
attempts and failed accepted prefixes are validated. Numerical qualification must
use developed lift cycles, separate mesh/geometry/time sensitivity and the cited
sampled reference; performance uses repeated interleaving and identical-schedule
replay, not weaker error gates. [Cylinder guide](../experiments/cylinder_ns/README.md)
and [validation](validation.md) separate achieved evidence from remaining limits.

2026-10-27 confirmed FEM extension: independent scikit-fem continuous Taylor–Hood
P2 velocity/P1 pressure on deterministic rectangular triangular meshes, homogeneous
no-slip, consistent mass/Riesz coefficient representation and antisymmetric convection.
Unpenalized direct shifted Stokes checks every weak continuity row and physical-integral
pressure gauge; physical gradient divergence is reported separately. All six shared
formats support fixed/prescribed steps. Dedicated run/load/reconstruct/readonly analysis
retains full coefficients and mesh/DOF metadata; legacy layouts remain unchanged.
That initial release excluded adaptivity and nonhomogeneous boundaries; the
2026-10-28 extension above added both. Its evidence did not establish a production
campaign or PDE time-order proof. See [FEM conventions](fem.md) and
[measured evidence](validation.md#2026-10-27-fem-release-verification).

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

### Original design (2026-09-20; historical scope)

Source: user-confirmed initial design. Later confirmed extensions above and below
supersede this section's original two-scheme and fixed-step scope.

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

2026-11-02 授权范围补充：Hele-Shaw可显式启用energy_stable_pressure入口，方向性动能牵引及SAV边界功采用相同transport；原natural默认及旧运行身份保留。可显式配置velocity_limit/scalar_limit作运行资格终止，不作状态裁剪。独立inlet_recovery从零初值先检查T=1.1，再从零运行Re_D=10000至T=20，生产使用既有backflow_damped出口。时间格式、线性系统及完整残差规则不改。

### Optional FEM scalability comparison (2026-11-02)

Explicit `stokes_backend=petsc`, `mass_backend=cg|metis` and `fem_load_mode=weak` are independent
optional controls for the shared closed/open/reaction FEM path. Defaults and old run
identities do not acquire these keys. Six time formats share one momentum adapter; weak
load storage changes no physical operator, trace, pressure scale or SAV root rule. All
original residual/continuity/gauge checks remain mandatory. Global LU fallbacks are counted
and disqualify an AMG-only performance claim. Comparison runs freeze both sources, actual
meshes and initial states; reports read explicit saved run IDs without launching flow solves.
Distributed FEM and unmeasured Hypre performance are outside this adapter's qualification.

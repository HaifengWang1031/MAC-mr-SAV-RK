# FEM: Taylor–Hood P2/P1

Independent fifth spatial Model: scikit-fem supplies affine triangular meshes,
bases, DOF maps and weak-form assembly; SciPy supplies sparse direct solves and
optional recycled-factor preconditioned solves of the identical shifted system.
The shared time schemes and stepping policies are unchanged; the integrator and
saved-run reader share pure compensated schedule/snapshot mapping arithmetic.
Locally installed and locked: scikit-fem 12.0.2, CPython 3.12.14, macOS arm64.

## Usage

```sh
uv sync --locked
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m experiments.fem_ns.run \
  --config experiments/fem_ns/configs/smoke.json --root /tmp/fem-smoke
# Use the exact printed directory, not a search for the latest run.
uv run --locked python -m experiments.fem_ns.analyze \
  /tmp/fem-smoke/runs/fem_ns/<run-id> --root /tmp/fem-smoke
```

All six names are supported: `imex_euler`, `imex_euler_mrsav`, `sdirk2`,
`sdirk2_mrsav`, `sdirk3`, `sdirk3_mrsav`. Use fixed `dt` or an explicit positive
`steps` sequence, not both. The bundled continuous polynomial Navier–Stokes MMS
is a tiny workflow example, not a convergence campaign. Embedded SDIRK2/3 formats
also support the existing I/PI driver (`mode=adaptive`, `initial_step`, `controller`,
optional SAV `scalar_atol`), without changing fixed record identities. See the
[experiment guide](../experiments/fem_ns/README.md) for configuration,
`run`, `load_fem`, `reconstruct`, `analyze`, reuse and `--rerun`.

The public numerical entry points are `solver.fem.Space` and `FEMModel`:

```python
from solver.fem import FEMModel, Space

model = FEMModel(Space(8, 4, lx=2., ly=1., quadrature_order=6), nu=.2)
projected = model.project(lambda x, y: (x*(2-x)*y*(1-y), 0.))
velocity = model.solve(projected, mass=1., viscosity=0.).velocity
initial = model.state(0., velocity, r=0.)
```

## Numerical conventions

- Domain `[0,lx] x [0,ly]`, deterministic `MeshTri.init_tensor`, continuous
  vector P2 velocity/scalar P1 pressure; all wall velocity DOFs, including edge
  nodes, are eliminated. The dimensionally underdetermined 1x1 mesh is rejected;
  singular mixed systems fail rather than being pressure-regularized.
- Internal velocity vectors contain free coefficients in library DOF ordering.
  `State.u/v` contain full scalar P2 component coefficients including wall zeros
  and edge nodes. Components use the library maps, **not contiguous halves** of
  a vector and not MAC face arrays. Pressure contains all P1 coefficients.
- `project` is a constrained consistent L2 projection, not point sampling or a
  divergence-free projection. The mass-only solve in the example imposes weak
  incompressibility. Analytic force loads likewise pass through a mass solve;
  `model.force(t)` must return velocity coefficients, never raw weak loads.
- With free velocity mass `M`, stiffness `A`, full weak divergence `B`, the
  Model uses `inner(a,b)=a^T M b`, `K=M^-1 A`, `G=M^-1(-B^T)`, `D=B`.
  Sparse mass solves are used, not lumping or a dense inverse. Shifted Stokes is
  `(mass M + viscosity A)u - B^T p = M rhs`, `Bu=0`, with zero pressure block
  and a physical-integral pressure gauge. Same-shift multiple RHS share a
  bounded factorization cache (default 4).
- Pressure zero mean means the basis-integral weighted domain integral is zero,
  **not** the arithmetic mean of coefficients. Every solve checks normalized
  weak momentum, **every** pressure-test continuity row and the integral gauge
  against the configured tolerance (default 1e-9). An additional mass-solved
  momentum-coefficient defect is checked. Default direct MMD ordering falls back
  to robust COLAMD when factorization or true checks fail. Optional `recycled`
  uses a bounded old LU only as a GMRES preconditioner for the exact new matrix;
  failed iteration/checks refresh directly. No step quantization or tolerance relaxation.
- Weak continuity `||Bu||_inf` and physical divergence are different diagnostics.
  `divergence_l2` integrates FE gradients; `divergence_inf` is their maximum at
  quadrature points, **not an analytic supremum**. A successful solve need not
  be pointwise divergence-free.
- Convection is the antisymmetric weak form
  `c(w;u,v)=((w.grad)u,v)/2-((w.grad)v,u)/2`, with `w=u` for `nonlinear(u)`.
  Both terms use the same quadrature; the weak load is mass-solved to coefficients.
  Thus nonlinear work cancels even for non-strongly-solenoidal discrete fields.
  `quadrature_order>=5` covers affine P2 mass and convection polynomials;
  nonpolynomial loads and manufactured forces need separate higher-order sensitivity
  checks. The workflow MMS force/test integrand reaches degree 15, so its default
  order 8 is not exact: versus order 18, projected-force mass-norm differences are
  `1.498927e-5` on 2x2/amplitude 1 and `2.299800e-10` on defaults 8x4/amplitude .1.
  Order 16 agrees within `2e-13` in both cases. Retained tests allow 1% around
  these order-8 measurements (absolute floor `1e-14`); defaults remain unchanged.
  These are force-projection sensitivities at t=0, not trajectory error bounds.
- Trial input is not mutated; failed integrations retain only accepted prefixes.
  Stage pressures retain registry multiplier semantics, including
  `euler_step_multiplier_mean_zero` for Euler and
  `incremental_stage_multiplier_mean_zero` for SDIRK. They are not demonstrated
  same-order physical endpoint-pressure approximations.

## Representation and analysis

Runs and reports use separate `fem_ns` namespaces. Mesh coordinates/connectivity,
elements/orders, scikit-fem version, DOF locations/element/component/free/boundary
maps, full P2 velocity coefficients and final accepted step's P1 stage pressures
are saved. Vertex-only output would lose the original field. Snapshots retain
velocity/r and actual times, **not pressure history**.

`load_fem` checks checksums, source/config identity, representation, coefficients,
constraints and trajectory metadata. Saved/installed scikit-fem versions must
match; no version migration is provided. Old source hashes may still be read;
reuse requires matching current config/source/environment and a complete valid
record. Rerun creates a distinct attempt without overwriting. Checksums establish
integrity, not authenticity against deliberate rewriting.

`reconstruct` and `analyze` only rebuild saved meshes/bases; neither constructs a
dynamical Model/backend nor integrates. Analysis requires an explicit run
directory and writes JSON/CSV/log reports. Contained run locators are relative
to the supplied project root; external inputs retain absolute locators.
The 17x13 display grid is derived
sampling, not the stored representation. Failed prefixes are labelled. Setup or
storage failures re-raise and may have no loadable results. Atomic rename does
not guarantee durable fsync recovery; interrupted records are not reused.

## Evidence and limits

[Release validation](validation.md#2026-10-27-fem-release-verification) records
independent asymmetric Stokes spatial errors, exact semidiscrete temporal
refinement, quadrature sensitivity, full regression/type status and a real tiny
CLI lifecycle check. Exact semidiscrete time references require no fine-step
reference; they do not establish continuous-PDE time order. Six runnable schemes,
small stage residuals and observed SAV rates are not general mrSAV order proofs.

The rectangular MMS remains homogeneous and has no curved/imported/periodic mesh,
other element pair, pressure stabilization or checkpoint continuation. The separate
[DFG2D-2 cylinder workflow](../experiments/cylinder_ns/README.md) adds verified saved
polygonal meshes, fixed nonzero inlet lifting and a natural outlet with **no pressure
gauge**. Its skew-volume convection includes boundary energy flux; its separately
reconstructed instantaneous pressure is not a stage multiplier. See
[ADR0001](adr/0001-open-flow-sav-boundary-work.md) for the open-boundary SAV work.

Cached library P2 quadrature data and owned value-checked field caches accelerate
convection; force callbacks evaluate once, and only the known MMS time modes are
preprojected. Repeated native-state performance results, including setup/observer
costs and common-schedule fidelity, are recorded separately from accuracy claims.
Legacy MAC/Legendre/Fourier/Channel APIs and records are unchanged.

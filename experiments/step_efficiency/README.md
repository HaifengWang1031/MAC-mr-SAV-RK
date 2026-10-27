# Warm per-step efficiency

Purpose: compare all seven current serial space/linear-solver combinations and six
time schemes after compilation/loading and model initialization. The main result
is CPU milliseconds per step, including one native diagnostic call. Separate step,
diagnostic and wall costs are saved too.

The centralized `configs/n128.json` uses N=128, a 1.3 × 0.8 domain, nu=0.1,
amplitude=0.1, gamma=1, nominal dt=0.002, five measured steps after two warmup
steps, and three interleaved repeats. Each case starts its measured interval again
at t=0. The cache contains the fixed-step factor; fixed timings exclude its
factorization. The seeded prescribed variable schedule has the same total time
and charges each new shift's factorization. It measures neither an adaptive
controller nor rejected steps. Storage is outside all timing intervals.

Smooth continuous MMS profiles come from the existing no-slip, channel and
periodic experiment modules. Constant force components are prepared in each
model's natural load space before measurement. Fourier force components are
transformed only during setup, rather than repeating FFTs of analytic samples at
every stage. State conversion, pressure output, full momentum/continuity checks,
nonlinear evaluation, forcing combinations and model diagnostics remain active.

N means cells per direction for MAC/Channel, collocation points for Fourier and
velocity modes for Legendre. Legendre's maximum polynomial degree is N+1 and its
nonlinear Gauss rule has at least ceil((3(N+1)+1)/2) nodes per direction. Fourier
uses strict 2/3 filtering. Boundary conditions and smooth profiles differ across
model families. Consequently this is a cost comparison at the requested N;
it does not rank equal accuracy or equivalent physical problems.

Run with the project environment and one thread in every numerical library:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
uv run python -m experiments.step_efficiency.run
```

Then analyze an explicit saved run, without constructing models or advancing time:

```bash
uv run python -m experiments.step_efficiency.analyze runs/step_efficiency/<run-id>
```

Runs contain config/manifest/source identity, progress log, incremental results
index and per-case NPZ files (final fields/stage pressures, actual times, stage
checks and every measured step's CPU/wall timings). Reports contain JSON, CSV and
two Markdown tables. Analysis verifies completeness and checksums, full residual
checks, cached/new factor counts and final-field agreement between the two
backends of each identical discretisation. These are short-run integrity and
backend consistency checks, not a new convergence proof or turbulence benchmark.

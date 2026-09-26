# No-slip Kolmogorov adaptive comparison

Purpose: compare I and PI controllers for SDIRK2, SDIRK2-mr-ccSAV, SDIRK3
and SDIRK3-mr-ccSAV at the same velocity error tolerance. On `(0,2*pi)^2`
with zero velocity at all walls, `f=(-sin(4*y),0)` has curl `4*cos(4*y)`.
The initial streamfunction is the periodic multimode sum multiplied by
`sin²(x/2) sin²(y/2)`, so it is wall-clamped; the MAC velocity is its discrete
curl and is exactly discretely divergence-free. `r0=0`, `nu=1/50`, `gamma=1000`.

The 2(1) SDIRK2 difference uses a first-order extrapolation of stage one and
therefore controller estimator order 2. The 3(2) SDIRK3 difference uses the
four-stage combination and estimator order 3. Both control the MAC velocity
discrete L² norm using `atol_velocity + rtol_velocity*max(norms)`.
The scalar tolerance from the periodic paper is **not** imposed: no validated
embedded scalar output exists yet. `r` is recorded separately. Rejected steps
never advance velocity, r or PI history. Hitting `min_step` with unacceptable
error marks failure; it never forces acceptance.

The earlier vorticity-controlled smoke records remain historical data. This
velocity-controlled configuration creates new run identities and the analysis
rejects the older batch format, so unlike-norm errors cannot be mixed.

`smoke.json` uses 24², three initial modes and T=0.04 solely to check the
workflow. It also runs each of the four schemes with `fixed_step=0.01`, the
adaptive maximum, as a direct same-problem control. `production.json` requests
128², ten modes, T=30, output every 0.1, `min_step=1e-5`, `max_step=1e-2`,
`fixed_step=1e-2`, `rtol_velocity=5e-5`, and adaptive initial step `1e-5`.
The smoke configuration uses the same adaptive initial step. This is an unrun
production proposal: adaptive direct Stokes solves can require a new sparse
factorization almost every step, and the fixed reference alone needs about
300,000 steps. Benchmark cost before launching it; do not infer 256² cost from
fixed-step tests.
The production log records progress every 1,000 accepted comparison steps and
every 10,000 reference steps; these intervals are set in `production.json`.

```sh
uv run python experiments/kolmogorov_adaptive/run.py \
  --config experiments/kolmogorov_adaptive/configs/smoke.json
uv run python experiments/kolmogorov_adaptive/analyze.py \
  --batch /absolute/path/to/batch.json
```

The run writes eight adaptive records, four fixed-step records, and a separate
fine-step SDIRK3 reference under `runs/kolmogorov_adaptive/<id>/`, each with
effective config, manifest, compressed arrays and failure status. The analysis
reads an explicit batch and generates **one six-panel PDF and PNG** covering
step sizes, cumulative accepted steps, CPU time, `|r|`, the velocity `L²`
embedded error divided by its absolute-plus-relative tolerance (adaptive only),
and relative reference-velocity error. The summary CSV retains final reference
error and CPU cost. A fixed-step run is not the reference run, and its CPU time
excludes reference computation and analysis. One tolerance cannot establish an
efficiency frontier; that requires a tolerance scan. Reference results at the
same grid are diagnostic, not a proof of accuracy until the reference step is
refined.

To compare each adaptive run separately against the same scheme at fixed steps
`0.005`, `0.001`, and `0.0005`, reuse a completed adaptive batch and run:

```sh
uv run python experiments/kolmogorov_adaptive/run_fixed_controls.py \
  --base-batch /absolute/path/to/original/batch.json
uv run python experiments/kolmogorov_adaptive/analyze_separate.py \
  --batch /absolute/path/to/new/batch.json
```

This adds twelve fixed-step records and makes eight separate six-panel PDF/PNG
figures, one for each scheme-controller pair. When the original reference step
is not finer than all fixed controls, it computes a finer SDIRK3 reference;
for the 24² smoke batch, this uses `0.0001`. The experiment remains a short
workflow check, not the unrun `T=30` production comparison.

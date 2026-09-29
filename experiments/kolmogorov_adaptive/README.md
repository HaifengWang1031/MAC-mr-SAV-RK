# No-slip Kolmogorov adaptive comparison

Purpose: compare I and PI controllers for SDIRK2, SDIRK2-mr-ccSAV, SDIRK3
and SDIRK3-mr-ccSAV at a common velocity error tolerance. On `(0,2*pi)^2`
with zero velocity at all walls, `f=(-sin(4*y),0)` has curl `4*cos(4*y)`.
The wall-clamped multimode streamfunction produces an exactly discretely
divergence-free MAC velocity. The physical parameters are `r0=0`, `nu=1/50`
and `gamma=1000`.

The 2(1) SDIRK2 pair uses estimator order 2 and the 3(2) SDIRK3 pair uses
estimator order 3. Both control the MAC velocity discrete L2 norm with
`atol_velocity + rtol_velocity*max(norms)`. The scalar `r` remains diagnostic:
no scalar tolerance is imposed because no validated embedded scalar output is
available. Rejected trials never advance velocity, `r`, or PI history.

## Current design

- The base campaign contains exactly eight adaptive runs and one fixed-step
  SDIRK3 reference. The known-invalid `tau=0.01` fixed control was removed.
- Both smoke and production configurations use the shift-reusable tensor Stokes
  backend. JIT warmup is measured before timed integration and excluded from
  per-attempt CPU time.
- Batch reuse requires identical effective configuration, source hash, complete
  status, and result checksum. Analysis applies the same consistency checks.
- Strict output nodes never introduce a trial below `min_step`. If a proposed
  step would leave a remainder below `min_step`, the distance is advanced
  by four equal accepted substeps. A small but admissible remainder is split
  into two equal steps to avoid an abrupt output-alignment step; roundoff-only
  endpoint gaps are
  snapped without a solve. The partition flags are stored in `results.npz`.
- The fine reference step and the velocity-reference error metric are unchanged.

`smoke.json` uses 24², three initial modes, and `T=0.04` only to verify the
workflow. `production.json` uses 128², ten modes, `T=30`, output every `0.1`,
`min_step=1e-5`, `max_step=1e-2`, `rtol_velocity=5e-5`, and reference step
`1e-4`. The production proposal is expensive: the reference alone requires
300,000 steps.

```sh
uv run python experiments/kolmogorov_adaptive/run.py \
  --config experiments/kolmogorov_adaptive/configs/smoke.json
uv run python experiments/kolmogorov_adaptive/analyze.py \
  --batch /absolute/path/to/batch.json
```

The base analysis generates one six-panel PDF/PNG and a summary CSV for step
sizes, accepted-step count, CPU time, `|r|`, normalized embedded error, and
relative reference-velocity error.

## Tolerance search

`tolerance_scan.json` defines a 128², `T=0.1` pilot over relative tolerances
`1e-3`, `5e-4`, `2e-4`, `1e-4`, and `5e-5`. It preserves
`atol/rtol=2e-4`, the existing `1e-4` reference step, and requires maximum
output velocity error at most `1e-4`. Every case is run three times in
forward/reverse/rotated order with all numerical-library thread counts fixed to
one. Model setup, numerical warmup, reference cost, integration CPU time, Stokes
work, rejection reasons, and boundary partitions are recorded separately.

```sh
uv run python experiments/kolmogorov_adaptive/search_tolerance.py \
  --config experiments/kolmogorov_adaptive/configs/tolerance_scan.json
```

The completed scan is stored at
`reports/timings/20260927T141542-82e76f6e-a87d30ee-adaptive-tolerance-scan/timings.json`.
All 120 adaptive timings completed and no trial was below `1e-5`. The common
tolerance satisfying the target for all eight scheme/controller pairs is still
`rtol=5e-5`, `atol=1e-8`; therefore `production.json` remains unchanged.
Per-case fastest admissible tolerances were `5e-5/2e-4` for SDIRK2 I/PI,
`5e-5/2e-4` for SDIRK2-mrSAV I/PI, `1e-4/5e-4` for SDIRK3 I/PI, and
`1e-4/2e-4` for SDIRK3-mrSAV I/PI. Using those heterogeneous tolerances would
reduce aggregate median pilot CPU time by 16.7%, but it is not appropriate for
the common-tolerance comparison. This short pilot does not establish long-time
`T=30` accuracy.

## Fixed controls

The separate fixed-step ladder uses `0.005`, `0.0025`, `0.001`, and `0.0005` for every
scheme. The coarsest value is below the observed plain IMEX-SDIRK2 stability
boundary; `0.01` is deliberately absent. Run it only from a complete,
source-matching adaptive base batch:

```sh
uv run python experiments/kolmogorov_adaptive/run_fixed_controls.py \
  --base-batch /absolute/path/to/original/batch.json
uv run python experiments/kolmogorov_adaptive/analyze_separate.py \
  --batch /absolute/path/to/new/batch.json
```

This phase writes sixteen fixed-step records and eight separate comparison
figures. If needed, it computes a finer SDIRK3 comparison reference. A fixed
control is not a reference run, and all sixteen controls must complete before
analysis.

## Pair-specific T=30 tuning

`configs/tuned_parameters.json` records the eight measured I/PI parameter sets.
The data, exact parameter values, and selection criterion are preserved in
`runs/kolmogorov_adaptive/tuning/selection-20260928T035927/selection.json`
and its `summary.md`. The criterion uses the **maximum** relative MAC velocity
L2 reference error across outputs from 0 to 30: each adaptive run must be
better than the same scheme at fixed `tau=0.0025`. This does not assert that
the adaptive curve is below that fixed curve at every output or at `T=30`.
Among measured cases passing the error and interior step-ratio checks, the
lower-CPU case is selected. These are empirical choices for this setup, not
globally optimal controller parameters.

The four `tau=0.005` fixed records and all eight tuned runs are included in
the separate six-panel figures. Regenerate them with:

```sh
uv run python experiments/kolmogorov_adaptive/analyze_tuned.py \
  --screen runs/kolmogorov_adaptive/tuning/selection-20260928T035927/selection.json
```

The selection record links to the existing fine reference and earlier fixed
controls, which were computed before the adaptive output-alignment change.
That change affects only adaptive stepping; each member retains its own source
hash and checksum, and the tuned analysis explicitly records the mixed-source
comparison.

## Natural adaptive nodes and optional reference errors

`configs/natural_step_reference.json` keeps the common absolute cap `0.005`
and controller factor range `[0.5, 1.2]`, with the pair-specific tolerances in
`configs/tuned_parameters.json`. This separate campaign uses no intermediate
snapshot requests: the adaptive solver chooses its accepted nodes naturally
and is required only to reach `T=30`.

Set `reference_diagnostic.enabled` to `true` to calculate relative MAC velocity
L2 errors at every accepted node. For each of the eight adaptive trajectories,
two independent fixed-SDIRK3 reference trajectories start from the same initial
state and advance on subdivisions of each accepted interval. Their maximum
substep sizes are `1e-4` and `5e-5`, with at least 8 and 16 subdivisions per
accepted interval. The reported adaptive error uses the finer reference; the
coarse/fine reference difference checks its temporal resolution. Both the
maximum-over-accepted-nodes and terminal errors are recorded. The maximum is
sampled on a different time grid for each method, so terminal errors are the
directly comparable common-time metric.

The reference calculation is an accepted-state diagnostic and never changes
the controller or its states. `adaptive_cpu_seconds` excludes all observer CPU
time, including reference solves and error evaluation;
`reference_cpu_seconds` counts the two reference solves;
`total_integration_cpu_seconds` includes both. Disable the option for an
application run without references. Error arrays are stored in each member's
`reference_error.npz`; the campaign manifest records verification against a
5% reference-refinement criterion. An unverified reference is reported as
such, not treated as an accurate result.

```sh
uv run python experiments/kolmogorov_adaptive/run_natural_step_error.py \
  --config experiments/kolmogorov_adaptive/configs/natural_step_reference.json
```

Add `--smoke` for a `T=0.2` check of one second-order and one third-order
pair. This is a new experiment and does not overwrite the older strict-output
campaigns or figures.

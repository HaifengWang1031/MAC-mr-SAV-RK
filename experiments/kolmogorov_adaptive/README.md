# No-slip Kolmogorov adaptive comparison

## 自然节点自适应/固定步对照成果

[中文论文实验描述](studies/adaptive_fixed/description.tex) ·
[材料状态与论文引入](studies/adaptive_fixed/README.md)

**Notebook 待补原始数据。** 本轮不以本地汇总表替代原始成员及双参考轨迹，
也不将已有生产报告的参考未完全验证状态写成已验证结论。

Purpose: compare I and PI controllers for SDIRK2, SDIRK2-mr-ccSAV, SDIRK3
and SDIRK3-mr-ccSAV at a common velocity error tolerance. On `(0,2*pi)^2`
with zero velocity at all walls, `f=(-sin(4*y),0)` has curl `4*cos(4*y)`.
The wall-clamped multimode streamfunction produces an exactly discretely
divergence-free MAC velocity. The physical parameters are `r0=0`, `nu=1/50`
and `gamma=1000`.

The 2(1) SDIRK2 pair uses estimator order 2 and the 3(2) SDIRK3 pair uses
estimator order 3. Both control the MAC velocity discrete L2 norm with
`atol_velocity + rtol_velocity*max(norms)`. These base campaigns leave scalar
`r` diagnostic. The separate joint-tolerance search below controls its defect
against the exact local decay equation. Rejected trials never advance velocity,
`r`, or PI history.

## Entry-point map

- Start with `run.py` + `analyze.py` and `configs/smoke.json` for the eight-member base campaign.
- `run_fixed_controls.py` + `analyze_separate.py` add controls to an explicitly selected, source-matching base.
- `search_tolerance.py` and `search_joint_tolerance.py` run distinct velocity-only and joint-control studies.
- `run_natural_step_error.py` uses the saved parameter values in `configs/tuned_parameters.json`; it does not require the old selection record.
- `run_adaptive_fixed.py`, `extend_adaptive_fixed.py`, `rerun_adaptive_fixed.py` and `replay_rejected.py` retain different parent/reuse/reference policies. Keep their explicit source requirements.
- Older selection-dependent workflows and engineering benchmarks are classified below; their source paths are retained.

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

The historical completed scan was recorded at
`reports/timings/20260927T141542-82e76f6e-a87d30ee-adaptive-tolerance-scan/timings.json`.
That artifact is absent from the current checkout; the following numbers are historical, not locally reverified results.
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

## Historical pair-specific T=30 tuning (external records required)

`configs/tuned_parameters.json` retains the eight measured I/PI parameter sets.
The original selection record was
`runs/kolmogorov_adaptive/tuning/selection-20260928T035927/selection.json`
with its `summary.md`; these records are absent from the current checkout.
`analyze_tuned.py`, `run_common_step_limits.py` with `configs/common_step_limits.json`,
and the fixed-control comparisons in `analyze_natural_partial.py` require an explicit old selection record and its linked runs.
Restore those original artifacts before using these historical workflows; do not substitute newly computed results.
The parameter file remains available to `run_natural_step_error.py` independently of those records. The criterion uses the **maximum** relative MAC velocity
L2 reference error across outputs from 0 to 30: each adaptive run must be
better than the same scheme at fixed `tau=0.0025`. This does not assert that
the adaptive curve is below that fixed curve at every output or at `T=30`.
Among measured cases passing the error and interior step-ratio checks, the
lower-CPU case is selected. These are empirical choices for this setup, not
globally optimal controller parameters.

The historical separate six-panel figures included four `tau=0.005` fixed records
and all eight tuned runs. After restoring the original selection and linked runs, regenerate them with:

```sh
uv run python experiments/kolmogorov_adaptive/analyze_tuned.py \
  --screen runs/kolmogorov_adaptive/tuning/selection-20260928T035927/selection.json
```

The selection record links to the existing fine reference and earlier fixed
controls, which were computed before the adaptive output-alignment change.
That change affects only adaptive stepping; each member retains its own source
hash and checksum, and the tuned analysis explicitly records the mixed-source
comparison.

## Engineering benchmarks and historical inputs

These are separate engineering studies, not additional physical campaign modes.
They remain at their original paths; classification here does not certify every benchmark on the current machine.

| Script | Question / required input |
| --- | --- |
| `benchmark_stokes_backends.py` | Direct versus tensor Stokes, fixed/prescribed steps, including backend setup |
| `benchmark_adaptive_backends.py` | Direct/tensor agreement of adaptive acceptance, step proposals and final velocity |
| `benchmark_matrix_free_stokes.py` | Sparse versus matrix-free operators in Stokes solves |
| `benchmark_matrix_free_integration.py` | Whether the matrix-free gain survives complete time stepping |
| `benchmark_transform_buffers.py` | Transform-buffer reuse with the operator path fixed |
| `benchmark_initial_vs_optimized.py` | Historical source comparison; requires an explicitly supplied preserved baseline checkout |

The last script's default `/private/tmp/mac-mrsav-baseline-13d8a26` does not exist in this checkout's environment.
Supply the original baseline with `--baseline-root`; do not replace it with current source and call that historical reproduction.
`benchmark_matrix_free_integration.py` imports helpers from `benchmark_stokes_backends.py`,
and `benchmark_transform_buffers.py` imports `benchmark_matrix_free_stokes.py`.
Any later archival must retain those dependencies and the exact source/environment records.

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
## Joint velocity/scalar tolerance search

`configs/joint_tolerance_scan.json` defines a separate SDIRK3-mrSAV I search on
the existing 128² Kolmogorov model (`nu=0.02`, `gamma=1000`, `T=30`). The local
scalar defect is `r_trial - exp(-gamma*h)*r_old`. Accept with
`max(E_u,E_r)<=1`, where `E_r=abs(defect)/atol_r`. The component step proposals
use estimator orders three and two respectively; PI, when used through the
solver API, keeps separate accepted component histories. A fixed `atol_r` is a
local tolerance, not a bound on the accumulated `abs(r)`.

The scan uses nine `(rtol_u,atol_r)` pairs at `atol_u=1e-8`, two additional
`atol_u` sensitivity cases, and one velocity-only baseline. All cases use the
same I-controller factors and step bounds (`1e-5 <= h <= 0.005`). Three
interleaved repeats report median, minimum and maximum integration CPU time.
Single-thread settings are applied before the CLI runs. JIT/stage warmup,
model setup, result writing and the two common reference solves are outside
the integration timing.

```bash
uv run experiments/kolmogorov_adaptive/search_joint_tolerance.py \
  --config experiments/kolmogorov_adaptive/configs/joint_tolerance_scan.json --pilot --report
uv run experiments/kolmogorov_adaptive/search_joint_tolerance.py \
  --config experiments/kolmogorov_adaptive/configs/joint_tolerance_scan.json --report
uv run experiments/kolmogorov_adaptive/analyze_joint_tolerance.py --campaign /absolute/path/campaign.json
```

The pilot has `T=0.2` and one repeat. It validates the workflow and estimates
cost; it misses the historical scalar-defect peak around `t=4.9`. Production
runs cover `T=30`. Uniform output nodes are reached by strict step clipping;
the comparison therefore concerns this output-aligned algorithm, not the old
natural-node trajectory. Maximum velocity error means the maximum **at these
saved outputs** (every `0.1` in production); `max_abs_r` uses all accepted nodes.

Common ordinary SDIRK3 references have steps `2e-4` and `1e-4`. A candidate
requires reference refinement differences no larger than 5% of both its
maximum and final velocity errors. Failure of this check yields no verified
recommendation for that candidate; it does not silently launch more reference
work. Reports preserve all measured cases, the three-objective CPU/velocity/r
Pareto set, and the fastest verified choices for several explicit velocity and
scalar error targets. These targets are comparison levels, not asserted error
bounds or a universal optimal tolerance.

Each launch creates a new directory under
`runs/kolmogorov_adaptive/joint_tolerance_scan/<id>/`, containing effective
configurations, source/environment metadata, manifests/checksums, attempts,
snapshot fields, component indicators/defects, stage-check maxima and timing
records. `run.log` owns progress reporting. Analysis consumes an explicit
campaign, writes to `reports/kolmogorov_adaptive/joint_tolerance_scan/<id>/`,
and never launches numerical computation. There is no checkpoint continuation
or automatic reuse; a restart creates a fresh campaign and preserves the old one.

## 256² natural-node adaptive versus fixed-step comparison

`configs/adaptive_fixed_256.json` centralizes the EIT-WSL comparison requested
on 2026-10-02. It uses the existing Kolmogorov model, not the independent
fixed-step lattice-convergence model: nu=0.02, f=(-sin(4y),0), gamma=1000,
the deterministic initial vertex streamfunction with epsilon=4 and mode radius
10, and homogeneous no-slip walls on (0,2pi)².

The adaptive SDIRK3-mrSAV I run uses atol_u=1e-8, rtol_u=2e-6, atol_r=1e-4,
initial step 1e-5, safety=0.9 and factors [0.5,1.2]. The lower safeguard is
1e-12; the upper bound is infinity (persisted as JSON null). Output requests
never clip adaptive steps: requests every 0.1 map to nearest natural nodes,
with the actual physical times saved. Eight same-scheme fixed controls have
steps 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001 and 0.0005. Production T=30 has three
interleaved single-thread repeats; the T=0.1 pilot has one repeat.

```bash
uv run experiments/kolmogorov_adaptive/run_adaptive_fixed.py \
  --config experiments/kolmogorov_adaptive/configs/adaptive_fixed_256.json --pilot --report
uv run experiments/kolmogorov_adaptive/run_adaptive_fixed.py \
  --config experiments/kolmogorov_adaptive/configs/adaptive_fixed_256.json --report
uv run experiments/kolmogorov_adaptive/analyze_adaptive_fixed.py \
  --campaign /absolute/path/campaign.json
```

Each launch creates `runs/kolmogorov_adaptive/adaptive_fixed/<id>/campaign.json`
and a flushed `run.log`. Integration CPU excludes model setup, JIT/stage
warmup, persistence and shared reference computation. Two ordinary SDIRK3
references (maximum steps 1e-4 and 5e-5) visit the union of actual saved sample
times; their field snapshots are saved once per campaign. No interpolation is
used. Refinement differences must be at most 5% of each candidate's maximum
and final sampled velocity errors to label its reference verified.

Member records retain attempts, indicators, accepted-node diagnostics,
requested/actual snapshots and timing. `stage_checks.npz` adds the final state,
accepted-stage scalar values, residuals, full divergence checks and all real
root candidates/residuals. Failure prefixes and failure times remain visible.
Analysis never computes new trajectories and writes PNG/PDF six-panel figures,
CSV summary and per-repeat error arrays under
`reports/kolmogorov_adaptive/adaptive_fixed/<id>/`. Panels show accepted step
sizes, accepted-step counts, integration CPU, |r|, relative L2 embedded difference,
and relative velocity L2 error at actual saved sample times. Representative
curves are chosen near median repeat CPU; the table retains all repeats.

Storage scales with saved snapshots, not every full state. At 256² one u/v
snapshot is about 1 MiB. At T=30 the 27 comparison members retain at most
roughly 8 GiB of raw snapshot fields; the union of at most 1,204 reference
times retains roughly 2.4 GiB across both references, before compression and
additional scalar/stage records. References are streamed. No checkpoint
continuation or automatic monitoring is provided; logs end with FINAL status
and, after plotting succeeds, REPORT. A pilot does not establish long-time
accuracy or a universal tolerance optimum.

The 2026-10-04 extension adds 0.01/0.005/0.0025 to the original completed
five-control campaign. `extend_adaptive_fixed.py` consumes an explicit parent
and checks every reused configuration, source/environment and checksum. Each
new step must divide output_every, so the existing two references cover all
new snapshots. It creates a derived campaign containing old member paths and
new computation records; parent data/references are never overwritten. A
preserved copy of the original solver/experiment source is used for numerical
execution, with the extension entrypoint separately checksummed. Current
plotting code handles the nine curves and records its own source identity.

```bash
uv run experiments/kolmogorov_adaptive/extend_adaptive_fixed.py \
  --parent-campaign /absolute/path/original/campaign.json --steps 0.01 0.005 0.0025 \
  --source-root /absolute/path/original-source-snapshot --output-root /absolute/path/project \
  --analysis-script /absolute/path/project/experiments/kolmogorov_adaptive/analyze_adaptive_fixed.py
```

Numerical source/environment must exactly match the parent; a source snapshot
must be prepared before editing the project. New runs use the parent's repeat
count, single-thread settings, model, initial condition and timing convention.
Failed controls retain their final state/stage diagnostics, but velocity-error
samples remain at saved reference-covered output times. The parent 0.0005
reference-unverified status is re-evaluated from the same two references by
analysis; no reference refinement is silently performed during extension.

The 2026-10-04 presentation update uses xlim=[0,T] on every panel ([0,30]
in production), omits the panel (f) legend and reference-refinement curve,
and retains the shared bottom legend. Refinement checks and error arrays
remain in reports; `analysis.json/plot_options` records these display choices.

Panel (e) shows `||u_high-u_embedded||_h / ||u_high||_h` at adaptive trials,
using the candidate endpoint norm for both accepted steps and rejected trials.
The unnormalized difference at accepted steps is recovered from `error_u` by
multiplying by `atol_velocity + rtol_velocity * max(sqrt(2*K_old), sqrt(2*K_new))`,
using the original MAC kinetic energy `K=||u||_h^2/2`. Alignment of accepted
attempts and energy nodes is checked. `adaptive-embedded_difference.npz`
retains times, recovered differences, trial norms, relative differences and
normalization scales. A zero difference at zero trial velocity is reported as
zero; a nonzero difference at zero trial velocity is undefined and rejected.
Fixed-step
embedded differences were not saved, so they are not plotted in (e).

For an early six-panel preview while the finer reference is still running,
analysis accepts `--preliminary-coarse`. It uses the completed first reference
at matching saved times for panel (f), marks the title PRELIMINARY, records
`preliminary_reference_unverified`, and leaves refinement differences unavailable
and all verification flags false. It never loads an unfinished reference or
interpolates missing fields. The ordinary final analysis still requires two
completed references. Computation continues independently of this preview.

Rejected trial differences can be supplied with an explicit, completed
`replay_rejected.py` diagnostic run. This separate entrypoint uses the original
source/environment and persisted checkpoints, replays recorded attempts up to
the last rejection in each checkpoint interval, and never advances state on a
rejected attempt. It checks velocity/scalar indicators and accepted-node kinetic
energies/scalars against original records (rtol=1e-9, atol=1e-11), retains the
unnormalized rejected differences, candidate velocity norms and checksums,
and leaves original data and
integration timing untouched. Analysis only loads this explicit diagnostic;
it plots red crosses at `attempt_t + attempt_h` with a `Rejected trials` legend.

```bash
uv run python experiments/kolmogorov_adaptive/replay_rejected.py \
  --member /absolute/path/to/adaptive-member --source-root /absolute/path/original-source \
  --output-root /absolute/path/project
uv run python experiments/kolmogorov_adaptive/analyze_adaptive_fixed.py \
  --campaign /absolute/path/campaign.json --rejected-replay /absolute/path/diagnostic-run
```

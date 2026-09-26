# No-slip forced NS reference-solution convergence

Purpose: compare SDIRK2, SDIRK2-mr-ccSAV, SDIRK3 and SDIRK3-mr-ccSAV on the
same non-manufactured solution. The unit-square no-slip flow has viscosity
`nu=0.001`, body force `(0, sin(2*pi*x))`, a deterministic clamped-beam
multimode initial velocity (seed 7, band 2--6, max speed 1), and `r0=0`.
The initial sampled velocity is discretely projected once. `gamma=1` is the
primary SAV case; `gamma=1000` is a separate future sensitivity study.

Production uses MAC 128², observations at 0.5, 1, 2, 4, nominal trial steps
`0.1*2**(-k)` for `k=0,1,2,2.2,2.4,2.6,2.8,3,4,5,6,7`, and a *common*
SDIRK3 reference at k=10. SDIRK3 k=11 independently checks the reference.
For fractional k, only the end of each observation interval is shortened to
hit the exact time; the nominal step is not silently rounded. The table uses
the nominal step ratio in the observed rate. A reference gap larger than 5%
of the smallest finite trial error marks the batch `reference_unresolved`.
This is a sensitivity check, not an error bound.

The four schemes share grid, physical parameters, initial seed and reference.
The analysis writes `errors.csv`, two SDIRK-pair tables for each of velocity
L² and H¹ seminorm, and a compiled PDF. Failed runs appear as NaN at times
without a snapshot. They do not contribute to rate calculations. The fixed
grid isolates temporal differences; spatial convergence needs a separate grid
study. The coarse-step behavior is empirical, not a stability theorem.

```sh
uv run python experiments/no_slip_reference_convergence/run.py \
  --config experiments/no_slip_reference_convergence/configs/smoke.json
uv run python experiments/no_slip_reference_convergence/run.py \
  --config experiments/no_slip_reference_convergence/configs/production.json
uv run python experiments/no_slip_reference_convergence/analyze.py \
  --batch /absolute/path/to/batch.json
```

The run script writes a batch manifest and line-buffered log under
`runs/no_slip_reference_convergence/batches/`. Each member uses the existing
`runs/forced_ns/<run-id>/` config, manifest, HDF5 and log. The report is under
`reports/no_slip_reference_convergence/<analysis-id>/`. Re-execution reuses
verified complete member runs unless `--rerun` is passed. No checkpoint resume
is claimed.

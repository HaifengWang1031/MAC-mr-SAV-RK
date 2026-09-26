# Four-scheme MAC temporal convergence study

Purpose: compare SDIRK2, SDIRK2-mr-ccSAV, SDIRK3 and SDIRK3-mr-ccSAV against the same known solution on a fixed MAC grid. The fixed controls are the unit no-slip square, `nu`, streamfunction amplitude, `gamma`, grid and final time. Only scheme and fixed time step vary. `configs/production.json` is the complete effective campaign plan: 128², T=2, tau=0.1*2^(-k), integer k=-1,...,8.

The vertex streamfunction is `psi=a(t) sin²(pi x) sin²(2 pi y)` with `a(t)=0.05 exp(-t)(1+0.25 sin(pi t))`. A discrete curl constructs W_h; the exact MAC velocity is `a(t) W_h`. The force is assembled from **the same MAC operators**: `f_h=a'(t)W_h+nu*a(t)*K_h W_h+a(t)²*N_h(W_h)`. This is a grid-consistent semidiscrete manufactured solution with pressure zero. It isolates time integration error; it is not a continuous-force spatial accuracy experiment.

`run.py` executes all four schemes at every k. Each member is saved by the existing workflow with its effective config, manifest, diagnostics, accepted prefix and error status. Failed/nonfinite members remain in the batch. `analyze.py` only reads the explicit batch, computes L² and H¹ errors and adjacent rates, writes CSV and one nine-column LaTeX table, and compiles `convergence.pdf`. Missing error entries are `NaN` and rates are `--`. A failed coarse step is not silently discarded; fine-step plateaus remain visible.

```sh
uv run --locked python experiments/manufactured_convergence/run.py --config experiments/manufactured_convergence/configs/production.json
```

The campaign log and batch manifest are under `runs/manufactured_ns_convergence/batches/<id>/`. Final artifacts are under `reports/manufactured_ns_convergence/<id>/`. `FINAL status=complete` means all 40 runs and the report completed; `complete_with_failed_trials` means the report exists but at least one solver run failed. Source code and environment are included in run identity. No automatic checkpoint continuation is implemented.

The production range is intentionally broad. Coarse k may be outside the asymptotic regime; fine third-order errors may approach the algebraic/roundoff floor. The result must be judged from observed errors and rates rather than assuming the formal orders at every k. A single 128² pilot at k=-1 had all four schemes complete, so NaN for ordinary SDIRK is a hypothesis, not an observed outcome in this manufactured case.

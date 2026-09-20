# Confirmed implementation specification
Source: user-confirmed design in the current task, 2026-09-20.

- Separate desktop project MAC-mr-SAV-RK; source Stokes_BC.py and Fourier project remain unchanged.
- Uniform rectangular MAC, homogeneous no-slip, mean-zero pressure, compatible divergence/gradient, positive viscous matrix, energy-compatible centered convection.
- Numba local kernels, no fastmath initially; direct sparse coupled Stokes backend and bounded factorization reuse; an interface for future iterative backends.
- Two schemes: ordinary IMEX-SDIRK2 and the notes' incremental SDIRK2-mr-ccSAV. Fixed and prescribed steps. Each SAV stage uses two common-matrix RHS solves and a cubic. Enumerate numerical real roots, validate residuals, choose minimum absolute r, save candidate roots/count/selection/residual. Handle degeneracy.
- State carries named u,v arrays and SAV r; separate incremental stage-pressure output.
- Experiments: Stokes manufactured (velocity/pressure spatial errors), unsteady NS manufactured (spatial and temporal error for both schemes), unforced decay (energy/divergence/r/large-step behavior). Temporal studies use same-grid fine-time references and further reference refinement.
- Agreed test seams: public MAC operators and boundary behavior; JIT kernels versus independent reference; Stokes/stage residuals and root behavior; integration/convergence; run/save/load/analyze/reuse/rerun behavior.
- runs/<experiment>/<run-id> has config.json, manifest.json, results.h5, run.log. Actual times, diagnostics, requested/actual snapshot mapping, final state and stage roots are retained. Analysis writes reports/<analysis>/<id> with analysis.json, figures, tables, analysis.log and explicit input run IDs.
- Identity includes effective config, code and environment. Reuse only verified complete compatible results. Preserve failures and accepted prefixes; rerun creates new identity attempt. No checkpoint continuation. Batches record config and members.
- Validate 32²,64²,128² and one non-square grid, algebraic structure, spatial/time convergence and workflows. Separate JIT warmup, factorization and integration timings.

Latest confirmed refinements: place modules directly under solver/ (no nextgen/). Manage the environment with uv, commit uv.lock, and use uv run for all commands.

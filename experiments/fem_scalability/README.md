# FEM scalability comparison

Frozen numerical sources and workers replay the same saved mesh and initial state.
`run.py` prepares/executes explicit campaigns; `report.py` reads saved records only.
Do not launch a time integration from an analysis command. Existing experiment
configurations may opt into the settings in the completed report's `presets.json`;
none is silently made a new default.

Completed run: `runs/fem_multigrid/20261102T083710`.
Human report: `reports/fem_multigrid/20261102T083710-final/summary.md`.
See `docs/fem-multigrid.md` for environment setup, baseline preparation prerequisites,
commands, pressure scaling/acceptance and measurement limitations.

Each worker records setup/online/process time, CPU, peak RSS, actual thread pools,
field arrays, stage roots/residuals, source/worker identities and optional-backend
metrics. A counted global LU fallback is not a qualified AMG performance sample.
Timeout repeats remain explicitly unrun rather than fabricating completion times.

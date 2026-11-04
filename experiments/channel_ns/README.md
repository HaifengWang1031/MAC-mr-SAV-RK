# Channel computation and analysis

Purpose: verify the periodic-x/no-slip-y MAC model, inspect pressure-driven channel
transients, and retain reproducible evidence independently of the old solver workflows.
`run.py` computes and saves; `analyze.py` reads explicit saved records and never advances
the solver. See [model conventions](../../docs/channel.md).

All six shared schemes support fixed/prescribed steps. Embedded I/PI adaptivity
is available for `sdirk2/sdirk2_mrsav` and `sdirk3/sdirk3_mrsav`; Euler has no
adaptive estimator. The default validation and benchmark groups below retain the
four SDIRK schemes; this subset does not limit the computation API.

```sh
uv run python -m experiments.channel_ns.run --config experiments/channel_ns/configs/smoke.json
uv run python -m experiments.channel_ns.run --config experiments/channel_ns/configs/adaptive.json
uv run python -m experiments.channel_ns.run --config experiments/channel_ns/configs/fft.json
uv run python -m experiments.channel_ns.analyze runs/channel_ns/<run-id>
uv run python -m experiments.channel_ns.analyze runs/channel_ns/<run-id> --plot
```

`--root` selects a storage root. `--rerun` creates a new attempt without overwriting
old results. Runs store effective config, source/environment identity, status/log,
real MAC fields, stage pressures/roots, diagnostics and requested/actual snapshots.
Adaptive attempts retain accepted and rejected trials. Failed runs keep their accepted
prefix and are never reused as complete runs. Representation tags explicitly describe
the periodic duplicated u endpoint; the reader validates checksums, fields and tags.

## Central parameters

| Parameter | Meaning |
|---|---|
| `nx,ny,lx,ly` | Uniform MAC cell counts and physical domain; ly is channel height |
| `nu` | Positive kinematic viscosity |
| `driving` | Uniform streamwise acceleration, equivalent to negative kinematic pressure gradient; signed or zero |
| `epsilon` | Nonnegative streamfunction scale in `perturbed`; peak perturbation speed divided by centreline speed in `ts_wave` |
| `amplitude` | Positive time-dependent streamfunction amplitude in MMS cases |
| `case` | `poiseuille`, `startup`, `perturbed`, `ts_wave`, `mms`, `discrete_mms` |
| `scheme,gamma,r0` | Existing scheme, SAV damping and initial scalar; MMS requires r0=0 |
| First-order choices | `imex_euler/imex_euler_mrsav`, fixed/prescribed steps; examples in `configs/imex_euler*.json` |
| `T,dt,steps` | Duration and fixed step or explicit positive sequence; use dt OR steps |
| `mode,initial_step,controller` | `adaptive` uses I/PI parameters instead of dt/steps |
| `scalar_atol` | Optional SDIRK3-mrSAV scalar local-defect tolerance; not a global error bound |
| `snapshots` | Requested physical times; actual accepted times are stored separately |
| `cache_size,stokes_tolerance` | LU cache capacity and full linear residual/divergence gate |
| `backend` | `direct` (default) or `fft`, both solving exactly the same MAC discretisation |
| `log_every` | Accepted-step logging interval |

Defaults are centralized in `run.DEFAULTS`; unknown keys are rejected. Forces and
initial conditions are centralized in `problems.py`. `backend=direct` is the default;
`backend=fft` accelerates the same MAC equations without filtering modes. Poiseuille starts at the exact discrete equilibrium;
startup starts from rest; perturbed adds a divergence-free vertex curl to that equilibrium.
MMS uses continuous derivatives, whereas discrete_mms is a fixed-grid temporal reference.

## Short verification study

```sh
uv run python -m experiments.channel_ns.validate
uv run python -m experiments.channel_ns.analyze runs/channel_ns_validation/<study-id> --validation
```

The default study varies dt on a fixed 12x10 grid for all four schemes and gamma=1/1000,
using 40/80/160 steps to enter the temporal asymptotic regime. It separately refines
16x8,32x16,64x32 steady continuous Stokes MMS and runs short startup/perturbed cases.
All temporal members and spatial fields are saved. Analysis rechecks member sources,
checksums and field-based errors before reporting rates. Configure a smaller matrix
with `--config`; this entry is a short validation study, not a production turbulence run.

Warmup, model setup, integration CPU/wall, LU factorization wall and solve wall times
are recorded separately. Factorization/solve costs are included in integration timing,
so they must not be added a second time.

## Same-discretisation timing comparison

```sh
uv run python -m experiments.channel_ns.benchmark
uv run python -m experiments.channel_ns.benchmark --config experiments/channel_ns/configs/benchmark.json
uv run python -m experiments.channel_ns.benchmark --analyze runs/channel_ns_benchmark/<study-id>
```

Defaults compare 32x24 and 64x48, four schemes, ten steps, and three interleaved repeats.
`configs/benchmark.json` additionally includes 128x96.
Set thread environment variables to one for a single-thread comparison. Every pair
uses the same effective physical/numerical case except backend. Analysis verifies
sources, checksums, actual times and final-field agreement, then reports median CPU
cost including setup, as well as integration CPU alone. A small grid or short run
can be dominated by mode-factorization overhead; speedups are measured, not assumed.
This comparison concerns two solvers for the same MAC equations, not spectral accuracy.

Optional PDF/PNG plots use XeLaTeX, amsmath and Times New Roman, matching the project's
academic figure convention. Analysis without `--plot` needs no TeX installation.

## Saved-field channel animation

```sh
uv run python -m experiments.channel_ns.run --config experiments/channel_ns/configs/animation.json
uv run python -m experiments.channel_ns.animate runs/channel_ns/<run-id>
```

The animation case is a pressure-driven 6x2 periodic channel, nu=0.04, driving=0.08,
with a small divergence-free streamfunction perturbation to discrete Poiseuille flow.
It uses 192x64 MAC cells, SDIRK3, the FFT/banded backend, dt=0.01 and T=8. Uniform
snapshot spacing 0.04 gives 201 saved frames. This is an illustrative relaxation
experiment, not a turbulence or mesh/time convergence study.

`animate` validates an explicitly selected complete run and never computes a PDE
solution. It exports MP4, looping GIF and initial/middle/final PNG previews under
`reports/channel_animation/<run-id>`. Colours have fixed limits across the movie.
The vorticity panel subtracts the exact discrete Poiseuille equilibrium; the profile
panel includes the continuous parabola. Tracers use bilinear MAC-face interpolation,
linear interpolation between saved times and RK2 with substeps no larger than 0.01.
They are visualisation trajectories, not a simulated passive-dye concentration.

`--fps`, `--particles` and `--dpi` control rendering only. `animation.json` records
actual/requested times, numerical checks, rendering settings, sources and output
checksums. Rendering requires ffmpeg, XeLaTeX and Times New Roman. Static labels are
rendered through PGF/XeLaTeX once, then combined with transparent changing data layers;
the clock uses the same system font. When available, pdffonts verifies embedded fonts.

## Tollmien--Schlichting instability

```sh
uv run python -m experiments.channel_ns.instability --config experiments/channel_ns/configs/instability.json
uv run python -m experiments.channel_ns.instability --analyze runs/channel_instability/<study-id>
uv run python -m experiments.channel_ns.animate runs/channel_ns/<nonlinear-run-id>
```

The default selects Umax=1, half-height=1, Re=30000 and alpha=0.8, so nu=1/30000,
F=2/30000 and Lx=2*pi/0.8. The primary 96x192 MAC run uses ordinary SDIRK3,
dt=0.05, T=400 and an initial peak perturbation speed 2.5% of Umax. The base
profile is the exact discrete pressure-driven Poiseuille equilibrium; no ongoing
wave forcing is applied. This Re definition uses centreline speed and half-height.

`ts_mode.py` constructs the fundamental linearised MAC eigenvector in the vertex-curl
divergence-free space. The pressure is eliminated only in this initial-condition
calculation; actual time stepping still uses the existing coupled Channel solver.
Eigenvalue, eigen-residual and unit seed velocity are retained in `initial_mode` in
the HDF5 record. An independent clamped Chebyshev Orr--Sommerfeld calculation matches
the published Re=10000, alpha=1 eigenvalue, lambda=0.003739670623-0.237526488820i.
See [Juniper, Hanifi and Theofilis, 2014](https://mpj1001.user.srcf.net/papers/AMR_Juniper_Hanifi_Theofilis_2014.pdf).

Before the nonlinear illustration, the study checks small-amplitude growth at
dt/2, on 128x256 cells, and in a stable Re=1000 control. It compares saved-field
growth with discrete eigenvalues and the independent continuous reference, using
explicit 2% gates plus full residual/divergence checks. Failed qualification does
not start the illustration. The nonlinear movie is not a convergence-certified
endpoint calculation or a claim about three-dimensional turbulence.

For TS records the renderer uses x-ct coordinates, with c from the initial eigenmode.
Colours still show laboratory-frame speed and perturbation vorticity. Particles are
advected using saved laboratory-frame velocities before the coordinate change;
instantaneous streamlines use the velocity relative to c. A dashed amplitude curve
shows the initial linear growth prediction, allowing nonlinear departures to be seen.

For a clearer view of the near-wall closed streamlines, export the lower 20% of the
channel in the vorticity panel (fivefold vertical zoom), while retaining the full
channel speed panel, mean profile and disturbance-amplitude history:

```sh
uv run python -m experiments.channel_ns.animate runs/channel_ns/<TS-run-id> \
  --near-wall --output reports/channel_instability_animation/<new-export-id>
```

`--near-wall` requires a saved TS-mode record. The zoom and the moving x-ct coordinate
are labelled; velocity colours remain laboratory-frame quantities. `--output` selects
a separate export directory so an existing movie can be retained. These options
only affect rendering, never the solver or stored velocity fields. The default
full-height movie layout remains available without these options.
The zoom uses a fixed 99th-percentile vorticity colour limit over the saved lower-wall
window, with colourbar extensions marking clipping. Its negative streamfunction
levels span the instantaneous near-wall minimum to zero, resolving closed contours
inside the vortex; contours remain instantaneous wave-frame streamlines.

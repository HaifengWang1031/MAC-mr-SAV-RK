# No-slip forced NS reference-solution convergence

## H-valued force campaign (2026-10-05)

Purpose: repeat the fixed-grid temporal comparison with a smooth steady force
in `V` and hence in `H`, and locate coarse-step failure or abrupt error changes
before selecting any fractional-k refinement. The independent configuration
`configs/h_force_256_gamma1000.json` compares all four SDIRK2/SDIRK3 methods at
integer `k=0,...,8`, with `tau=0.1*2**(-k)`. The new force is
`f=((1-cos(x))*sin(y), -sin(x)*(1-cos(y)))`, with amplitude `C=1`, on
`(0,2*pi)^2`. It has continuous divergence zero and full zero boundary trace.
It is sampled analytically at MAC faces, without normalization or an extra
projection. These samples also have discrete divergence zero up to roundoff
on the square grid; this exact cancellation is not asserted for unequal mesh
spacings.

The existing 256² lattice initial data, `nu=0.001`, `gamma=1000`, `r0=0`,
`T=4`, observations at `0.5,1,2,4`, and direct Stokes tolerance `1e-9` remain
the controls. All 36 trial runs and both common SDIRK3 references (`k=10,11`)
must use the new force and the same recorded source. Previous-force solutions
cannot be reused. Outputs are four error/rate tables in one PDF plus CSV, with
failed prefixes and reference sensitivity recorded as in the existing workflow.
Integer results will determine later refinement intervals; none are prescribed
from the previous-force transitions. This checks temporal differences at a fixed
grid, not spatial convergence or an unconditional PDE convergence theorem.

Run on the user-specified MAC MINI checkout after port mapping is enabled:

```sh
uv run python experiments/no_slip_reference_convergence/run.py \
  --config experiments/no_slip_reference_convergence/configs/h_force_256_gamma1000.json
```

The MAC MINI numerical checkout must be inspected before applying the force-only
change; unrelated local solver developments must not be copied into that baseline.

## Current third-order campaign (2026-10-02)

`configs/sdirk3_256_gamma1000.json` compares only SDIRK3 and SDIRK3-mr-ccSAV.
The domain is `(0,2*pi)^2`, with homogeneous no-slip walls, MAC 256²,
`nu=0.001`, steady force `(0,sin(x))`, `gamma=1000`, and `r0=0`.
The initial streamfunction is

\[
\psi_0(x,y)=4\sin^2(x/2)\sin^2(y/2)
\sum_{k_1,k_2\in\mathbb Z,\;0<k_1^2+k_2^2\le100}
\frac{[\cos(k_1x)+\sin(k_1x)][\cos(k_2y)+\sin(k_2y)]}
{(k_1^2+k_2^2)^{3/2}}.
\]

`lattice.py` includes all 316 signed integer modes, including axis modes.
There is no seed, random phase, or velocity normalization. Vertex samples of
psi produce face velocities by the discrete curl `u=delta_y psi`,
`v=-delta_x psi`; wall streamfunction values are exactly zero. This gives
discrete divergence zero up to roundoff, without a projection that changes
the prescribed amplitude. Tangential walls use the existing homogeneous
no-slip MAC closure. Spatial differentiation remains an approximation of
the prescribed continuous velocity.

Trial target steps are `0.1*2**(-k)` for integer `k=0,...,8` only, with
observations at `0.5,1,2,4` and final time 4. Both schemes use the same ordinary
SDIRK3 reference at `k=10` (`tau=0.00009765625`); `k=11`
(`tau=0.000048828125`) checks reference sensitivity. The existing sparse direct
Stokes backend and `stokes_tolerance=1e-9` are retained. A reference gap above
5% of the smallest positive finite trial error marks `reference_unresolved`;
absence of such an error also fails the check. This is a numerical sensitivity
criterion, not an error bound.

```sh
uv run python experiments/no_slip_reference_convergence/run.py \
  --config experiments/no_slip_reference_convergence/configs/sdirk3_256_gamma1000.json
```

The 18 trial runs and two references automatically produce
`tables/errors.csv`, `tables/sdirk3_L2.tex`, `tables/sdirk3_H1.tex`, and
`convergence.pdf` after computation. H1 denotes the MAC viscous-operator
seminorm. CSV rows retain run IDs, failure status and maximum endpoint/stage
scalar values. Failed trials retain their valid prefixes, and unavailable
snapshots appear as NaN. Reference failure prevents a validated comparison.
The batch log ends with `FINAL status=...` and the explicit report path.

## Preserved legacy campaign

`configs/production.json` compares SDIRK2, SDIRK2-mr-ccSAV, SDIRK3 and SDIRK3-mr-ccSAV on the
same non-manufactured solution. The unit-square no-slip flow has viscosity
`nu=0.001`, body force `(0, sin(2*pi*x))`, a deterministic clamped-beam
multimode initial velocity (seed 7, band 2--6, max speed 1), and `r0=0`.
The initial sampled velocity is discretely projected once. `gamma=1` is the
primary SAV case in this legacy configuration.

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

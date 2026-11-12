# Embedded SDIRK2(1) and SDIRK3(2) I/PI controllers

`integrate_adaptive` uses the shared Model interface for MAC, Legendre, Fourier,
Channel and FEM. It advances the main endpoint state from any of the four
embedded SDIRK schemes. All six schemes support fixed/prescribed stepping, but
`IMEXEuler` and `IMEXEulerMRSAV` return no embedded velocity; neither an Euler
embedded estimator nor step doubling is implemented. Public callers construct
IController/PIController directly and match the estimator order in the
[registry](../schemes/registry.py). Optional local experiment configurations use
`experiments/controllers.py:make_controller` for their own policy/serialization;
that adapter is not required or shipped with a core-only checkout.

## Embedded velocities and velocity error

The embedded combinations reuse already computed stages:

| Main schemes | Main/embedded velocity order | `estimator_order` |
| --- | --- | --- |
| `SDIRK2`, `SDIRK2MRSAV` | 2/1 | 2 |
| `SDIRK3`, `SDIRK3MRSAV` | 3/2 | 3 |

Here `estimator_order` is the local power of the main-minus-embedded difference,
not the order of the embedded velocity. With `eta=1-1/sqrt(2)`, SDIRK2 uses

\[
\bar u^{n+1}=(1-1/\eta)u^n+U_{n,1}/\eta.
\]

SDIRK3 uses

\[
\bar u^{n+1}=\tfrac23u^n-\tfrac{128}{99}U_{n,1}
                 +\tfrac{50}{33}U_{n,2}+\tfrac19U_{n,3}.
\]

By default the model inner product defines the velocity norm. The optional
`error_norm` callback replaces it consistently for the difference and both states:

\[
E_u=\frac{\|u^{n+1}-\bar u^{n+1}\|}
 {\mathrm{atol}+\mathrm{rtol}\max(\|u^n\|,\|u^{n+1}\|)}.
\]

With local power `q=estimator_order`, the default I update is
`0.9 * E_u**(-1/q)`, clipped to `[0.2, 2]`; zero error gives the maximum factor.
PI uses I for its first accepted step, then
`0.9 * max(E_u,1e-14)**(-0.7/q) * max(previous_E_u,1e-14)**(0.4/q)`
with the same clips. These factors and the PI exponents/floor are configurable.
The low-level API defaults to `estimator_order=3`; direct SDIRK2 callers must set
`estimator_order=2` on their controller.

## Optional scalar local-defect control

Passing `ScalarErrorControl(atol, gamma)` adds

\[
E_r=\frac{|r^{n+1}-e^{-\gamma h}r^n|}{\mathrm{atol}_r},
\qquad E=\max(E_u,E_r).
\]

This scalar indicator uses local power 2, its own accepted PI error history, and
the same configured factor bounds. Velocity and scalar each propose a step factor;
the driver takes the smaller. The default remains velocity-only control.
Experiment eligibility is local: periodic and Channel expose scalar control for
`sdirk3_mrsav`; FEM, cylinder and Hele–Shaw accept either adaptive SAV format.
This defect against `r'=-gamma*r` does not bound accumulated scalar error or `|r|`.

## Acceptance, rejection and output nodes

Under the default `min_step_policy='fail'`, a trial must have `E<=1`, finite state,
error and diagnostics, and successful stage/root checks. A rejected trial restarts
from the last accepted state. Its shrink factor uses the current-trial I estimate
and is capped at 0.8; rejected trials never update accepted PI histories or the
committed SAV scalar. Repeated failure returns the accepted prefix with
`status='failed'`.

The last step is clipped to `initial.t + T - current.t`. This terminal alignment
step may be shorter than `min_step`, including a roundoff-sized remainder, and
still undergoes the usual stage and error checks. After acceptance, its state,
time record, endpoint snapshot and callbacks use exactly `initial.t + T`.

`strict_snapshots=False` maps requests to nearest accepted nodes, taking the earlier
node on an exact tie. With `strict_snapshots=True`, steps align to increasing output
requests; near an internal output boundary the driver may partition the remaining
interval into two or four steps. Internal output alignment does not permit ordinary
steps below `min_step`.

Explicit `min_step_policy='force'` clamps ordinary proposals to `min_step` and
permits a finite over-tolerance trial exactly at that floor to be accepted. Stage
failures, nonfinite states/errors and invalid diagnostics still fail; terminal
alignment steps shorter than the floor retain the error check. Attempts preserve
the original error and acceptance decision. This option does not establish that
the requested local tolerance is met, and experiment exposure is configuration-specific.

## API and saved records

```python
from solver.adaptivity import PIController, integrate_adaptive
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV

result = integrate_adaptive(
    model, SDIRK3MRSAV(), initial, T=2.0, initial_step=0.01,
    controller=PIController(atol=1e-8, rtol=1e-4, max_step=0.1),
    snapshots=[0.0, 1.0, 2.0],
)
```

`result.attempts` records physical start time, trial step, normalized error,
previous accepted error, chosen factor, controller type, acceptance and rejection
reason for every trial. `result.times`, diagnostics,
stages and snapshots contain accepted states only. `T` is elapsed duration;
snapshot times are absolute. `integrate_adaptive` itself returns an in-memory result;
experiment runners write manifests, HDF5 and accepted/rejected attempts through the
shared record layer. Saved workflows exist for MAC adaptive studies, periodic,
Channel, FEM, cylinder and Hele–Shaw; Legendre uses the same API in dedicated studies.
See [architecture](../../docs/architecture.md) for shared record interfaces;
`experiments/README.md` is an optional local index of specialized entry points/readers.

### Optional bounded history and natural-node continuation

`retain_history=False` keeps only the initial/latest time and latest trial,
diagnostics and stage summaries. Use `result.accepted_steps` / `attempted_steps`,
not list lengths, for invocation counters. Explicit snapshot requests still retain
their requested states; production streaming therefore passes no snapshot list.
`on_attempt(record)` receives every trial; `on_advance(result)` runs after an
accepted step and the next step proposal are committed. Observer errors stop with
a failed prefix; they never authorize accepting a numerically rejected trial.

`result.continuation` is an `AdaptiveContinuation` with proposed `step`, previous
joint/velocity/scalar accepted errors and cumulative accepted/attempted counters.
Pass it together with the same accepted state and controller/scalar settings to
continue a natural-node trajectory. Strict-output boundary partitions are not
restartable through this interface (`strict_snapshots=True` is rejected on resume).
Experiments own identity checks, atomic storage, sampler cursors and log offsets;
the solver does not itself publish checkpoints. Default history retention and
numerical formulas remain unchanged. The first production use was explicitly
requested without further tests; new streaming/restart equivalence tests exist
but were not run before launch, so no new numerical qualification is implied.

The main-minus-embedded difference estimates the lower-order local error. It is
not a proved local or global error bound for the main output. Variable-step SAV
accuracy, PDE convergence and long-time stability require their own assumptions
and evidence; implemented adaptivity alone does not establish them.

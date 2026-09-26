# IMEX-SDIRK3(2) I and PI controllers

`integrate_adaptive` advances the third-order endpoint state from either `SDIRK3`
or `SDIRK3MRSAV`. The embedded second-order velocity is formed from the already
computed stages,

\[
\bar u^{n+1}=\tfrac23u^n-\tfrac{128}{99}U_{n,1}
                 +\tfrac{50}{33}U_{n,2}+\tfrac19U_{n,3}.
\]

The MAC/model inner product gives the norm of the difference. A trial is accepted
when its normalized error is at most one and its state and diagnostics are finite.
The I controller uses `0.9 * error**(-1/3)`, clipped to `[0.2, 2]`. The PI
controller uses the same rule for its first accepted step and thereafter uses
`0.9 * error**(-0.7/3) * previous_accepted_error**(0.4/3)` with the same clips.
Both use the current-trial I estimate to shrink a rejected step; rejected trials
do not change the previous accepted error. In either case, rejected trials
are retried from the last accepted state. The main SAV scalar is committed only
with the accepted third-order state. Failed stage/root solves trigger a smaller
trial; repeated failure returns the accepted prefix with `status='failed'`.

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
snapshot times are absolute. This solver-level API does not yet write run manifests
or HDF5 experiment records. The error difference is an estimator for the embedded
order-two output, not a proven local error bound for the order-three output.
Variable-step mr-ccSAV accuracy and long-time stability need separate validation.

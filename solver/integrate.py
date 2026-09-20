"""Common fixed/prescribed-step driver, bounded state storage and failure prefixes."""
from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from .core import State, Stage, Scheme, History
from .mac.grid import Array
from .mac_ns import MACNavierStokes

@dataclass
class Result:
    final: State
    times: list[float]
    diagnostics: list[dict[str,float]]
    stages: list[list[dict]] = field(default_factory=list)
    final_stages: list[Stage] = field(default_factory=list)
    snapshot_requests: list[float] = field(default_factory=list)
    snapshot_times: list[float] = field(default_factory=list)
    snapshots: list[State] = field(default_factory=list)
    status: str = 'complete'
    error: str = ''
    seconds: float = 0.

def step_sizes(T: float, *, dt: float | None = None, steps: list[float] | None = None) -> Array:
    if not np.isfinite(T) or T<=0: raise ValueError('T must be positive')
    if (dt is None)==(steps is None): raise ValueError('Specify exactly one of dt or steps')
    if steps is not None:
        sequence=np.asarray(steps,dtype=float)
        if sequence.ndim!=1 or sequence.size==0 or not np.isfinite(sequence).all() or np.any(sequence<=0):
            raise ValueError('Steps must be finite and positive')
        if not np.isclose(np.sum(sequence),T,rtol=1e-12,atol=1e-14*T):
            raise ValueError('Prescribed steps must sum to T')
        return sequence
    assert dt is not None
    if not np.isfinite(dt) or dt<=0: raise ValueError('dt must be positive')
    ratio=T/dt
    nearest=round(ratio)
    if nearest>0 and abs(ratio-nearest)<=1e-12*max(1,ratio):
        return np.full(nearest,dt)
    count=int(np.floor(ratio))
    return np.r_[np.full(count,dt),T-count*dt]

def integrate(model: MACNavierStokes, scheme: Scheme, initial: State, steps: list[float] | Array,
              snapshots: list[float] | None = None) -> Result:
    sequence=np.asarray(steps,dtype=float)
    if sequence.ndim!=1 or sequence.size==0 or np.any(sequence<=0) or not np.isfinite(sequence).all():
        raise ValueError('Invalid step sequence')
    nodes=np.r_[initial.t,initial.t+np.cumsum(sequence)]
    requests=[] if snapshots is None else list(snapshots)
    if any(not np.isfinite(t) or t<initial.t or t>nodes[-1]+1e-12 for t in requests):
        raise ValueError('Snapshot time outside integration interval')
    indices=[int(np.argmin(np.abs(nodes-t))) for t in requests]
    kept: dict[int,State]={}
    if 0 in indices: kept[0]=initial
    history=History(initial)
    result=Result(initial,[initial.t],[model.diagnostics(initial)])
    start=perf_counter()
    for k,dt in enumerate(sequence):
        try:
            trial=scheme.step(model,history.state,float(dt))
            if not np.isfinite(model.vector(trial.state)).all() or not np.isfinite(trial.state.r):
                raise FloatingPointError('Nonfinite trial state')
            diagnostic=model.diagnostics(trial.state)
            if not all(np.isfinite(v) for v in diagnostic.values()): raise FloatingPointError('Nonfinite diagnostic')
            history.state=trial.state
            history.accepted_steps+=1
            result.final=trial.state
            result.times.append(trial.state.t)
            result.diagnostics.append(diagnostic)
            result.final_stages=trial.stages
            result.stages.append([{'r':s.r,'candidates':s.candidates,'root_residuals':s.root_residuals,
                                   'root_count':len(s.candidates),'residual':s.residual,
                                   'scalar_residual':s.scalar_residual,'divergence_inf':s.divergence_inf}
                                  for s in trial.stages])
            if k+1 in indices: kept[k+1]=trial.state
        except Exception as error:
            result.status='failed'
            result.error=f'{type(error).__name__}: {error}'
            break
    for request,index in zip(requests,indices):
        if index in kept:
            result.snapshot_requests.append(request)
            result.snapshot_times.append(kept[index].t)
            result.snapshots.append(kept[index])
    result.seconds=perf_counter()-start
    return result

"""I and PI controllers for the four-stage IMEX-SDIRK3(2) embedded pair."""
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import perf_counter, process_time
from typing import Any, Literal

import numpy as np

from ..core import Scheme, stage_diagnostics
from ..integrate import Result
from ..model import Model


@dataclass(frozen=True)
class IController:
    atol: float = 1e-8
    rtol: float = 1e-4
    safety: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 2.0
    min_step: float = 1e-10
    max_step: float = float('inf')
    max_rejections: int = 12
    max_steps: int = 1_000_000
    estimator_order: int = 3

    def __post_init__(self) -> None:
        if not np.isfinite([self.atol, self.rtol, self.safety, self.min_factor,
                            self.max_factor, self.min_step]).all():
            raise ValueError('Controller parameters must be finite')
        if self.atol < 0 or self.rtol < 0 or self.atol + self.rtol == 0:
            raise ValueError('At least one tolerance must be positive')
        if not (0 < self.safety < 1 and 0 < self.min_factor < 1 < self.max_factor):
            raise ValueError('Invalid I-controller factors')
        if self.min_step <= 0 or np.isnan(self.max_step) or self.max_step < self.min_step:
            raise ValueError('Invalid step bounds')
        if self.max_rejections < 0 or self.max_steps <= 0:
            raise ValueError('Invalid trial limits')
        if type(self.estimator_order) is not int or self.estimator_order < 2:
            raise ValueError('Invalid estimator order')

    def accepted_factor(self, error: float, previous_error: float | None) -> float:
        """The first accepted step uses the local order-three I estimate."""
        if error == 0:
            return self.max_factor
        return float(np.clip(self.safety*error**(-1/self.estimator_order),
                             self.min_factor, self.max_factor))


@dataclass(frozen=True)
class PIController(IController):
    proportional: float = 0.7
    integral: float = 0.4
    error_floor: float = 1e-14

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite([self.proportional, self.integral, self.error_floor]).all() \
                or self.proportional <= self.integral or self.integral < 0 \
                or not 0 < self.error_floor < 1:
            raise ValueError('Invalid PI-controller parameters')

    def accepted_factor(self, error: float, previous_error: float | None) -> float:
        if previous_error is None:
            return super().accepted_factor(error, previous_error)
        current = max(error, self.error_floor)
        previous = max(previous_error, self.error_floor)
        factor = self.safety * current**(-self.proportional/self.estimator_order) \
                 * previous**(self.integral/self.estimator_order)
        return float(np.clip(factor, self.min_factor, self.max_factor))


@dataclass(frozen=True)
class ScalarErrorControl:
    """Local defect against r'=-gamma*r, scaled by a fixed absolute tolerance.

    This controls the first-order auxiliary update, whose local defect is O(h^2).
    It does not impose an absolute bound on the accumulated scalar error |r|.
    """
    atol: float
    gamma: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.atol, self.gamma]).all() or self.atol <= 0 \
                or self.gamma < 0:
            raise ValueError('Invalid scalar error controls')

    def defect(self, old_r: float, trial_r: float, step: float) -> float:
        return float(trial_r-np.exp(-self.gamma*step)*old_r)


@dataclass(frozen=True)
class AdaptiveContinuation:
    """Natural-node restart data; no accepted PI component is reset on restart."""
    step: float
    previous_error: float | None = None
    previous_u: float | None = None
    previous_r: float | None = None
    accepted_steps: int = 0
    attempted_steps: int = 0


@dataclass
class AdaptiveResult(Result):
    """Result plus one record per trial, including rejected trials."""
    attempts: list[dict[str, float | bool | str]] = field(default_factory=list)
    accepted_steps: int = 0
    attempted_steps: int = 0
    continuation: AdaptiveContinuation | None = None


def _error(model: Model, old: Any, high: Any, low: Any, controller: IController,
           error_norm: Callable[[Any], float] | None) -> float:
    difference = model.combine((1., high), (-1., low))
    squared = (error_norm(difference)**2 if error_norm is not None else
               model.inner(difference, difference))
    old_squared = (error_norm(old)**2 if error_norm is not None else model.inner(old, old))
    high_squared = (error_norm(high)**2 if error_norm is not None else model.inner(high, high))
    if not np.isfinite([squared, old_squared, high_squared]).all() or min(
            squared, old_squared, high_squared) < 0:
        return float('inf')
    scale = controller.atol + controller.rtol * max(np.sqrt(old_squared), np.sqrt(high_squared))
    return float(np.sqrt(squared) / scale)


def integrate_adaptive(model: Model, scheme: Scheme, initial: Any, T: float, initial_step: float,
                       *, controller: IController = IController(),
                       snapshots: list[float] | None = None,
                       error_norm: Callable[[Any], float] | None = None,
                       strict_snapshots: bool = False,
                       progress: Callable[[int, float, float], None] | None = None,
                       on_accept: Callable[[Any], None] | None = None,
                       scalar_control: ScalarErrorControl | None = None,
                       min_step_policy: Literal['fail', 'force'] = 'fail',
                       retain_history: bool = True,
                       continuation: AdaptiveContinuation | None = None,
                       on_attempt: Callable[[dict[str, float | bool | str]], None] | None = None,
                       on_advance: Callable[[AdaptiveResult], None] | None = None) -> AdaptiveResult:
    """Advance the order-three state; rejected trials never change the accepted state.

    T is a duration from initial.t. Snapshot requests are absolute physical times
    and map to the nearest accepted node (earlier node wins an exact tie).
    The terminal step reaches initial.t+T exactly and may be below min_step;
    it still passes the same error and stage checks as every other trial.
    With scalar_control, accept max(E_u,E_r)<=1 and take the smaller component
    step proposal. Velocity and scalar use their own orders and PI histories.
    Explicit force policy clamps ordinary proposals to min_step and permits
    finite over-tolerance trials exactly at that floor. Nonfinite states/errors,
    failed stage solves and diagnostics are still rejected. Shorter terminal
    alignment steps retain their error check; default fail policy is unchanged.
    """
    if min_step_policy not in ('fail', 'force'):
        raise ValueError('Unknown min_step_policy')
    if not np.isfinite([T, initial_step, initial.t]).all() or T <= 0 or initial_step <= 0:
        raise ValueError('T and initial_step must be finite and positive')
    if initial_step < controller.min_step:
        raise ValueError('initial_step is below min_step')
    end = initial.t + T
    requests = [] if snapshots is None else list(snapshots)
    if any(not np.isfinite(t) or t < initial.t or t > end for t in requests):
        raise ValueError('Snapshot time outside integration interval')
    if strict_snapshots and requests != sorted(set(requests)):
        raise ValueError('Strict snapshot times must be distinct and increasing')
    best = [(abs(t-initial.t), initial) for t in requests]
    result = AdaptiveResult(initial, [initial.t], [model.diagnostics(initial)])
    if continuation is not None:
        if strict_snapshots:
            raise ValueError('Continuation supports natural nodes, not strict snapshot partitions')
        if not np.isfinite(continuation.step) or not controller.min_step <= continuation.step <= controller.max_step:
            raise ValueError('Invalid continuation step')
        history_values = [v for v in (continuation.previous_error, continuation.previous_u,
                                      continuation.previous_r) if v is not None]
        if any(not np.isfinite(v) or v < 0 for v in history_values) or \
                continuation.accepted_steps < 0 or continuation.attempted_steps < continuation.accepted_steps:
            raise ValueError('Invalid continuation history')
    saved_history = continuation or AdaptiveContinuation(min(initial_step, controller.max_step))
    step = saved_history.step
    result.continuation = saved_history
    rejections = 0
    previous_error = saved_history.previous_error
    previous_u = saved_history.previous_u
    previous_r = saved_history.previous_r
    scalar_controller = replace(controller, estimator_order=2)
    output_index = 0
    boundary_step: float | None = None
    boundary_steps_remaining = 0
    if strict_snapshots and requests and requests[0] == initial.t:
        output_index = 1
    start = perf_counter()
    cpu_start = process_time()
    while result.final.t < end:
        if saved_history.accepted_steps + result.accepted_steps >= controller.max_steps:
            result.status, result.error = 'failed', 'Maximum accepted steps exceeded'
            break
        remaining = end-result.final.t
        time_tolerance = 8*np.finfo(float).eps*max(abs(end), abs(result.final.t), 1.)
        proposed_step = min(step, remaining)
        step = proposed_step
        boundary_partition = False
        output_truncated = False
        if strict_snapshots and output_index < len(requests):
            distance_to_output = requests[output_index]-result.final.t
            if abs(distance_to_output) <= time_tolerance:
                output_index += 1
                continue
            if distance_to_output < 0:
                result.status, result.error = 'failed', 'Strict snapshot time was crossed'
                break
            if boundary_steps_remaining:
                assert boundary_step is not None
                step = min(boundary_step, distance_to_output)
                boundary_partition = True
            else:
                output_truncated = proposed_step >= distance_to_output
                remainder = distance_to_output-proposed_step
                if 0 < remainder < controller.min_step and \
                    distance_to_output/4 >= controller.min_step:
                    boundary_step = distance_to_output/4
                    boundary_steps_remaining = 4
                    step = boundary_step
                    boundary_partition = True
                elif 0 < remainder < 0.5*proposed_step and \
                    distance_to_output/2 >= controller.min_step:
                    # Split the last two steps evenly instead of creating a tiny
                    # exact-output-alignment step after an ordinary accepted step.
                    boundary_step = distance_to_output/2
                    boundary_steps_remaining = 2
                    step = boundary_step
                    boundary_partition = True
                else:
                    step = min(proposed_step, distance_to_output)
        terminal_step = step == remaining
        force_floor = min_step_policy == 'force' and step == controller.min_step
        if step < controller.min_step and not terminal_step:
            result.status, result.error = 'failed', 'Step below min_step'
            break
        reason = ''
        error = float('inf')
        error_u = float('inf')
        error_r = float('nan')
        scalar_defect = float('nan')
        trial = None
        try:
            trial = scheme.step(model, result.final, step)
            if trial.embedded_velocity is None:
                raise ValueError('Scheme does not provide an embedded velocity')
            high = model.vector(trial.state)
            if terminal_step:
                trial.state = model.state(end, high, trial.state.r)
            if not np.isfinite(high).all() or not np.isfinite(trial.embedded_velocity).all() \
                    or not np.isfinite(trial.state.r):
                raise FloatingPointError('Nonfinite trial state')
            error_u = _error(model, model.vector(result.final), high,
                             trial.embedded_velocity, controller, error_norm)
            error = error_u
            if scalar_control is not None:
                scalar_defect = scalar_control.defect(result.final.r, trial.state.r, step)
                error_r = abs(scalar_defect)/scalar_control.atol
                error = max(error_u, error_r) if np.isfinite(error_r) else float('inf')
            if not np.isfinite(error):
                reason = 'nonfinite_error'
            elif error > 1. and not force_floor:
                reason = 'error_exceeded'
            else:
                diagnostic = model.diagnostics(trial.state)
                if not all(np.isfinite(value) for value in diagnostic.values()):
                    raise FloatingPointError('Nonfinite diagnostic')
        except (FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
            reason = f'{type(exc).__name__}: {exc}'
        if reason:
            # Rejection depends only on the current trial; accepted PI history is
            # retained but never updated by a rejected trial.
            factor = min(IController.accepted_factor(controller, error_u, None), 0.8)
            if scalar_control is not None:
                scalar_error = error_r if np.isfinite(error_r) else float('inf')
                factor = min(factor, IController.accepted_factor(
                    scalar_controller, scalar_error, None))
        else:
            factor = controller.accepted_factor(error_u, previous_u)
            if scalar_control is not None:
                factor = min(factor, scalar_controller.accepted_factor(error_r, previous_r))
        result.attempts.append({'t': float(result.final.t), 'step': float(step),
                                'error': float(error), 'accepted': not bool(reason),
                                'error_u': float(error_u), 'error_r': float(error_r),
                                'scalar_defect': float(scalar_defect),
                                'reason': reason, 'factor': float(factor),
                                'boundary_partition': boundary_partition,
                                'output_truncated': output_truncated,
                                'previous_accepted_error': (float('nan') if previous_error is None
                                                            else previous_error),
                                'previous_accepted_error_u': (float('nan') if previous_u is None
                                                              else previous_u),
                                'previous_accepted_error_r': (float('nan') if previous_r is None
                                                              else previous_r),
                                'controller': type(controller).__name__,
                                'elapsed_seconds': perf_counter()-start-result.observer_wall_seconds,
                                'cpu_seconds': process_time()-cpu_start-result.observer_cpu_seconds})
        result.attempted_steps += 1
        if not retain_history:
            result.attempts[:] = result.attempts[-1:]
        if on_attempt is not None:
            observer_wall_start, observer_cpu_start = perf_counter(), process_time()
            try:
                on_attempt(result.attempts[-1])
            except Exception as exc:
                result.status, result.error = 'failed', f'Trial observer failed: {type(exc).__name__}: {exc}'
                break
            finally:
                result.observer_cpu_seconds += process_time()-observer_cpu_start
                result.observer_wall_seconds += perf_counter()-observer_wall_start
        if reason:
            if boundary_partition:
                boundary_step = None
                boundary_steps_remaining = 0
            rejections += 1
            if rejections > controller.max_rejections:
                result.status, result.error = 'failed', reason or 'Maximum rejections exceeded'
                break
            proposed = step*factor
            if proposed < controller.min_step:
                if min_step_policy == 'force' and step > controller.min_step and not terminal_step:
                    step = controller.min_step
                    continue
                result.status, result.error = 'failed', reason
                break
            step = proposed
            continue
        assert trial is not None
        rejections = 0
        previous_error = error
        previous_u = error_u
        if scalar_control is not None:
            previous_r = error_r
        result.final = trial.state
        if strict_snapshots and output_index < len(requests) and \
                abs(trial.state.t-requests[output_index]) <= 8*np.finfo(float).eps*max(1.,abs(trial.state.t)):
            output_index += 1
        result.accepted_steps += 1
        result.times.append(trial.state.t)
        result.diagnostics.append(diagnostic)
        result.final_stages = trial.stages
        result.stages.append([stage_diagnostics(stage) for stage in trial.stages])
        if not retain_history:
            result.times[:] = [initial.t, trial.state.t]
            result.diagnostics[:] = result.diagnostics[-1:]
            result.stages[:] = result.stages[-1:]
        for index, request in enumerate(requests):
            distance = abs(request-trial.state.t)
            if distance < best[index][0]:
                best[index] = (distance, trial.state)
        # Commit restart context for this accepted state before its observers run.
        if boundary_partition:
            boundary_steps_remaining -= 1
            if boundary_steps_remaining:
                assert boundary_step is not None
                step = boundary_step
            else:
                boundary_step = None
                step = min(step*factor, controller.max_step)
        else:
            step = min(step*factor, controller.max_step)
        if min_step_policy == 'force' and not boundary_steps_remaining:
            step = max(step, controller.min_step)
        result.continuation = AdaptiveContinuation(
            step, previous_error, previous_u, previous_r,
            saved_history.accepted_steps+result.accepted_steps,
            saved_history.attempted_steps+result.attempted_steps)
        if on_accept is not None:
            observer_wall_start = perf_counter()
            observer_cpu_start = process_time()
            try:
                on_accept(trial.state)
            except Exception as exc:
                result.status = 'failed'
                result.error = f'Accepted-state observer failed: {type(exc).__name__}: {exc}'
                break
            finally:
                result.observer_cpu_seconds += process_time()-observer_cpu_start
                result.observer_wall_seconds += perf_counter()-observer_wall_start
        if progress is not None:
            progress(result.accepted_steps, trial.state.t,
                     perf_counter()-start-result.observer_wall_seconds)
        if on_advance is not None:
            observer_wall_start, observer_cpu_start = perf_counter(), process_time()
            try:
                on_advance(result)
            except Exception as exc:
                result.status, result.error = 'failed', f'Advance observer failed: {type(exc).__name__}: {exc}'
                break
            finally:
                result.observer_cpu_seconds += process_time()-observer_cpu_start
                result.observer_wall_seconds += perf_counter()-observer_wall_start
    for request, (_, state) in zip(requests, best):
        result.snapshot_requests.append(request)
        result.snapshot_times.append(state.t)
        result.snapshots.append(state)
    result.seconds = perf_counter()-start-result.observer_wall_seconds
    return result

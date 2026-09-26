"""I and PI controllers for the four-stage IMEX-SDIRK3(2) embedded pair."""
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter, process_time
from typing import Any

import numpy as np

from ..core import Scheme
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


@dataclass
class AdaptiveResult(Result):
    """Result plus one record per trial, including rejected trials."""
    attempts: list[dict[str, float | bool | str]] = field(default_factory=list)


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
                       progress: Callable[[int, float, float], None] | None = None) -> AdaptiveResult:
    """Advance the order-three state; rejected trials never change the accepted state.

    T is a duration from initial.t. Snapshot requests are absolute physical times
    and map to the nearest accepted node (earlier node wins an exact tie).
    """
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
    step = min(initial_step, controller.max_step)
    rejections = 0
    previous_error: float | None = None
    output_index = 0
    if strict_snapshots and requests and requests[0] == initial.t:
        output_index = 1
    start = perf_counter()
    cpu_start = process_time()
    while result.final.t < end:
        if len(result.times)-1 >= controller.max_steps:
            result.status, result.error = 'failed', 'Maximum accepted steps exceeded'
            break
        remaining = end-result.final.t
        if remaining <= 8*np.finfo(float).eps*max(abs(end), abs(result.final.t), 1.):
            break
        proposed_step = min(step, remaining)
        step = proposed_step
        alignment_sliver = False
        if strict_snapshots and output_index < len(requests):
            distance_to_output = requests[output_index]-result.final.t
            step = min(step, distance_to_output)
            alignment_sliver = (0 < distance_to_output < controller.min_step
                                and proposed_step >= controller.min_step)
        if step < controller.min_step and remaining > controller.min_step \
                and not alignment_sliver:
            result.status, result.error = 'failed', 'Step below min_step'
            break
        reason = ''
        error = float('inf')
        trial = None
        try:
            trial = scheme.step(model, result.final, step)
            if trial.embedded_velocity is None:
                raise ValueError('Scheme does not provide an embedded velocity')
            high = model.vector(trial.state)
            if not np.isfinite(high).all() or not np.isfinite(trial.embedded_velocity).all() \
                    or not np.isfinite(trial.state.r):
                raise FloatingPointError('Nonfinite trial state')
            error = _error(model, model.vector(result.final), high,
                           trial.embedded_velocity, controller, error_norm)
            if not np.isfinite(error):
                reason = 'nonfinite_error'
            elif error > 1.:
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
            factor = min(IController.accepted_factor(controller, error, None), 0.8)
        else:
            factor = controller.accepted_factor(error, previous_error)
        result.attempts.append({'t': float(result.final.t), 'step': float(step),
                                'error': float(error), 'accepted': not bool(reason),
                                'reason': reason, 'factor': float(factor),
                                'previous_accepted_error': (float('nan') if previous_error is None
                                                            else previous_error),
                                'controller': type(controller).__name__,
                                'elapsed_seconds': perf_counter()-start,
                                'cpu_seconds': process_time()-cpu_start})
        if reason:
            rejections += 1
            if rejections > controller.max_rejections or step*factor < controller.min_step:
                result.status, result.error = 'failed', reason or 'Maximum rejections exceeded'
                break
            step *= factor
            continue
        assert trial is not None
        rejections = 0
        previous_error = error
        result.final = trial.state
        if strict_snapshots and output_index < len(requests) and \
                abs(trial.state.t-requests[output_index]) <= 8*np.finfo(float).eps*max(1.,abs(trial.state.t)):
            output_index += 1
        result.times.append(trial.state.t)
        result.diagnostics.append(diagnostic)
        result.final_stages = trial.stages
        result.stages.append([{'r': s.r, 'candidates': s.candidates,
                               'root_residuals': s.root_residuals,
                               'root_count': len(s.candidates), 'residual': s.residual,
                               'scalar_residual': s.scalar_residual,
                               'divergence_inf': s.divergence_inf,
                               'continuity_residual': s.continuity_residual}
                              for s in trial.stages])
        for index, request in enumerate(requests):
            distance = abs(request-trial.state.t)
            if distance < best[index][0]:
                best[index] = (distance, trial.state)
        if progress is not None:
            progress(len(result.times)-1, trial.state.t, perf_counter()-start)
        step = min((proposed_step if alignment_sliver else step)*factor,
                   controller.max_step)
    for request, (_, state) in zip(requests, best):
        result.snapshot_requests.append(request)
        result.snapshot_times.append(state.t)
        result.snapshots.append(state)
    result.seconds = perf_counter()-start
    return result

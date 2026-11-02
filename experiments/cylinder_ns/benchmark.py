"""Phase-free developed lift cycles and explicit, non-certified project reference gates."""
import numpy as np
from scipy.signal import find_peaks
from numpy.typing import NDArray

Array = NDArray[np.float64]
REFERENCE = {'drag_min': 3.164264028, 'drag_max': 3.2273925778,
             'lift_min': -1.0212876662, 'lift_max': .98656549629,
             'strouhal': .1/.33125,
             'pressure_min': 2.42036197503, 'pressure_max': 2.50702497068}
REFERENCE_SOURCE = {
    'url': 'https://wwwold.mathematik.tu-dortmund.de/~featflow/media/dfg_bench2new_2d/20141124_dfg_bench2_official.zip',
    'sha256': '08db59605b2f60efc512d2bbc79628ea5d1cc720c0986938a16920884a0215be',
    'members': ['bdforces_q2_lv6_dt4', 'pointvalues_q2_lv6_dt4'],
    'limits': 'Published periodic numerical samples, not certified acceptance intervals.'}


def _extrema(time: Array, values: Array, *, minimum: bool, prominence: float = 0.) -> list[tuple[float, float]]:
    indices, _ = find_peaks(-values if minimum else values, prominence=prominence)
    result = []
    for index in indices:
        t, value = float(time[index]), float(values[index])
        local = time[index-1:index+2]-t
        width = float(np.max(np.abs(local)))
        if min(abs(local[0]), abs(local[2])) > 1e-4*width:
            a, b, c = np.polyfit(local/width, values[index-1:index+2], 2)
            if (a > 0. if minimum else a < 0.):
                vertex = -b/(2*a)
                if local[0]/width <= vertex <= local[2]/width:
                    t, value = t+float(vertex*width), float(a*vertex**2+b*vertex+c)
        result.append((t, value))
    return result


def _statistics(time: Array, values: Array, start: float, end: float) -> tuple[float, float, float]:
    mask = (time > start) & (time < end)
    selected_time = np.r_[start, time[mask], end]
    selected_values = np.r_[np.interp(start, time, values), values[mask], np.interp(end, time, values)]
    extrema = [value for minimum in (False, True)
               for t, value in _extrema(time, values, minimum=minimum) if start <= t <= end]
    endpoints = [float(selected_values[0]), float(selected_values[-1])]
    bounds = extrema+endpoints+selected_values.tolist()
    return min(bounds), max(bounds), float(np.trapezoid(selected_values, selected_time)/(end-start))


def cycle_statistics(time: Array, drag: Array, lift: Array, pressure: Array, *,
                     mean_velocity: float, diameter: float) -> dict:
    arrays = [np.asarray(v, dtype=float) for v in (time, drag, lift, pressure)]
    time, drag, lift, pressure = arrays
    if any(v.ndim != 1 or v.shape != time.shape or not np.isfinite(v).all() for v in arrays) \
            or np.any(np.diff(time) <= 0.) or not np.isfinite([mean_velocity, diameter]).all() \
            or min(mean_velocity, diameter) <= 0.:
        raise ValueError('Invalid cylinder cycle samples/normalization')
    if len(time) < 5:
        return {'developed': False, 'cycles': 0, 'reason': 'Insufficient samples for three complete lift cycles'}
    late = lift[time >= time[0]+.5*(time[-1]-time[0])]
    prominence = max(1e-4, .2*float(np.ptp(late)))
    minima = _extrema(time, lift, minimum=True, prominence=prominence)
    maxima = _extrema(time, lift, minimum=False, prominence=prominence)
    if len(minima) < 4:
        return {'developed': False, 'cycles': max(0, len(minima)-1),
                'reason': 'Fewer than four resolved lift minima'}
    minima = minima[-4:]
    amplitudes = []
    for first, second in zip(minima, minima[1:]):
        peaks = [value for t, value in maxima if first[0] < t < second[0]]
        if not peaks:
            return {'developed': False, 'cycles': 0, 'reason': 'Unresolved lift-cycle maximum'}
        amplitudes.append(max(peaks)-min(first[1], second[1]))
    periods = np.diff([t for t, _ in minima])
    period = float(np.mean(periods))
    period_variation = float(np.ptp(periods)/period)
    amplitude_variation = float(np.ptp(amplitudes)/np.mean(amplitudes))
    developed = period_variation < .005 and amplitude_variation < .01
    start, end = minima[0][0], minima[-1][0]
    result = {'developed': developed, 'cycles': 3, 'window': [start, end],
              'period': period, 'periods': periods.tolist(),
              'strouhal': diameter/(mean_velocity*period), 'mean_velocity': mean_velocity, 'diameter': diameter,
              'lift_peak_to_peak_per_cycle': amplitudes,
              'period_relative_range': period_variation, 'amplitude_relative_range': amplitude_variation,
              'reason': ('Stable final three complete lift cycles' if developed else
                         'Period range/mean >= .5% or peak-to-peak range/mean >= 1%')}
    for name, values in (('drag', drag), ('lift', lift), ('pressure', pressure)):
        low, high, mean = _statistics(time, values, start, end)
        result.update({name+'_min': low, name+'_max': high, name+'_mean': mean,
                       name+'_midrange': .5*(low+high), name+'_peak_to_peak': high-low})
    return result


def reference_comparison(statistics: dict, cfg: dict, *, complete: bool) -> dict:
    canonical = (cfg.get('nu') == .001 and cfg.get('inlet_max') == 1.5 and cfg.get('r0') == 0.
                 and cfg.get('outlet_treatment', 'natural') == 'natural')
    errors = {key: abs(statistics[key]-value)/abs(value) for key, value in REFERENCE.items() if key in statistics}
    tolerances = {key: .03 if key.startswith('pressure') else .02 for key in REFERENCE}
    passed = (complete and canonical and bool(statistics.get('developed')) and len(errors) == len(REFERENCE)
              and all(errors[key] <= tolerances[key] for key in REFERENCE))
    return {'passed': passed, 'complete_run': complete, 'canonical_physics': canonical, 'applicable': canonical,
            'reference': REFERENCE, 'reference_source': REFERENCE_SOURCE,
            'relative_errors': errors, 'project_relative_tolerances': tolerances,
            'limits': 'Project gates on CD/CL extrema and lift Strouhal (2%), pressure extrema (3%); '
                      'not published certified tolerances. Mesh/geometry and temporal sensitivity '
                      'must be verified separately before overall benchmark qualification.'}

"""Read-only correlation estimates and pre-registered finite-window screens.

Geyer IPS pairs Gamma_k=rho[2k]+rho[2k+1] give the statistical inefficiency
``g = -1 + 2*sum(Gamma_k)``. Thus ``tau_int=interval*g/2`` (not half the pair
sum). We use the biased FFT autocovariance and conservatively cap ESS at N.
These are operational diagnostics, never a proof of stationarity.
"""
from typing import Any

import numpy as np

CONSTANT_RTOL = 1e-13
MIN_SAMPLES = 16
CORRELATION_METHOD = 'geyer_ips_biased_fft_v2'


def _pair_sums(rho: np.ndarray) -> tuple[float, int]:
    """Sum consecutive positive even/odd autocorrelation pairs starting at lag 0."""
    total, pairs = 0., 0
    for k in range(len(rho)//2):
        pair = float(rho[2*k]+rho[2*k+1])
        if not np.isfinite(pair) or pair <= 0:
            break
        total += pair
        pairs += 1
    return total, pairs


def integrated_correlation_time(values: np.ndarray, interval: float) -> dict[str, Any]:
    """IPS integrated time in sample-time units, ESS, and the first 128 ACF lags.

    Constant, nonfinite, short or untruncated estimates have explicit statuses;
    none supplies a fabricated finite error. ESS is capped at the sample count,
    equivalent to flooring tau at interval/2 for anticorrelated/noisy estimates.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(interval) or interval <= 0:
        raise ValueError('Require a scalar series and a positive sampling interval')
    n = len(x)
    base: dict[str, Any] = {'method': CORRELATION_METHOD, 'n_samples': n,
                            'tau_int': None, 'n_eff': None, 'pairs_used': 0, 'acf': []}
    if n < MIN_SAMPLES or not np.isfinite(x).all():
        return {**base, 'status': 'insufficient', 'reason': f'need >= {MIN_SAMPLES} finite samples'}
    scale = float(np.max(np.abs(x)))
    if scale == 0. or np.ptp(x) <= CONSTANT_RTOL*scale:
        return {**base, 'status': 'constant', 'reason': 'series has no resolvable variance'}
    centred = (x-x.mean())/scale
    size = 1 << int(np.ceil(np.log2(2*n)))
    spectrum = np.fft.rfft(centred, size)
    # Biased covariance: all lags have the same N denominator (cancels in rho).
    acov = np.fft.irfft(spectrum*np.conj(spectrum), size)[:n]
    if not np.isfinite(acov).all() or acov[0] <= 0:
        return {**base, 'status': 'insufficient', 'reason': 'nonpositive/invalid covariance'}
    rho = acov/acov[0]
    total, pairs = _pair_sums(rho)
    raw_g = 2*total-1
    base.update(pairs_used=pairs, acf=rho[:128].tolist())
    if not np.isfinite(raw_g) or raw_g <= 0 or pairs == 0 or pairs == len(rho)//2:
        return {**base, 'status': 'insufficient', 'reason': 'IPS did not yield a positive truncated estimate'}
    g = max(1., raw_g)
    return {**base, 'status': 'ok', 'tau_int': float(interval*g/2),
            'n_eff': float(n/g), 'duration': float(n*interval),
            'observed_span': float((n-1)*interval), 'ess_capped_at_samples': raw_g < 1., 'reason': ''}


def interval_quality(correlation: dict[str, Any], block: float, n_blocks: int) -> dict[str, Any]:
    tau, n_eff = correlation.get('tau_int'), correlation.get('n_eff')
    ratio = block/tau if tau else None
    checks = {'block_at_least_10_tau_int': ratio is not None and ratio >= 10.,
              'at_least_20_blocks': n_blocks >= 20,
              'n_eff_at_least_100': n_eff is not None and n_eff >= 100.}
    return {'block_over_tau_int': ratio, 'requirements': checks,
            'qualified': all(checks.values()), 'note': 'operational requirement, not a stationarity proof'}


def _series(times: np.ndarray, values: np.ndarray, start: float) -> tuple[np.ndarray, np.ndarray, float]:
    t, x = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if t.ndim != 1 or t.shape != x.shape or len(t) < 2 or \
            not np.isfinite(t).all() or not np.isfinite(x).all() or \
            not np.isfinite(start) or np.any(np.diff(t) <= 0) or \
            not np.allclose(np.diff(t), t[1]-t[0], rtol=1e-7, atol=1e-10):
        raise ValueError('Require matching finite uniformly sampled scalar series')
    interval = float(t[1]-t[0])
    keep = t > start+1e-10
    return t[keep], x[keep], interval


def _standard_error(values: np.ndarray, interval: float) -> dict[str, Any]:
    correction = integrated_correlation_time(values, interval)
    stderr = (float(np.std(values, ddof=1)/np.sqrt(correction['n_eff']))
              if correction['status'] == 'ok' else None)
    return {'mean': float(np.mean(values)), 'stderr': stderr, 'n': len(values),
            'correlation': correction, 'stderr_reason': correction['reason']}


def half_window_comparison(times: np.ndarray, values: np.ndarray, *, start: float) -> dict[str, Any]:
    """Compare disjoint halves; constant halves may establish drift, never an ESS."""
    t, x, interval = _series(times, values, start)
    if len(x) < 8:
        return {'status': 'insufficient', 'reason': 'too few samples after burn-in'}
    middle = len(x)//2
    first, second = _standard_error(x[:middle], interval), _standard_error(x[middle:], interval)
    out: dict[str, Any] = {'first': first, 'second': second, 'split_time': float(t[middle]),
                           'difference': float(second['mean']-first['mean'])}
    # Deterministic constants can flag a jump, but are not granted statistical qualification.
    if first['stderr'] is None or second['stderr'] is None:
        if first['correlation']['status'] == second['correlation']['status'] == 'constant':
            out.update(status='constant', combined_stderr=None, difference_over_combined_stderr=None,
                       reason='constant halves: no sampling uncertainty estimated')
        else:
            out.update(status='insufficient', reason='a half window has no resolvable error')
        return out
    combined = float(np.hypot(first['stderr'], second['stderr']))
    full = float(np.mean(x))
    out.update(status='ok', combined_stderr=combined,
               difference_over_combined_stderr=float(out['difference']/combined) if combined > 0 else None,
               difference_over_abs_mean=abs(out['difference']/full) if full else None)
    return out


def screen(times: np.ndarray, values: np.ndarray, *, start: float,
           injection_reference: float | None = None, storage_rate: float | None = None,
           storage_fraction_limit: float = .02, drift_fraction_limit: float = .05,
           drift_stderr_limit: float = 2.) -> dict[str, Any]:
    """One-window flux/storage screen; drift scales with injection, not a near-zero flux."""
    if not np.isfinite([storage_fraction_limit, drift_fraction_limit, drift_stderr_limit]).all() or \
            min(storage_fraction_limit, drift_fraction_limit, drift_stderr_limit) <= 0:
        raise ValueError('Screen thresholds must be positive and finite')
    reference_ok = injection_reference is not None and np.isfinite(injection_reference) and injection_reference > 1e-30
    reference = float(injection_reference) if reference_ok and injection_reference is not None else None
    result: dict[str, Any] = {'thresholds': {'storage_fraction_limit': storage_fraction_limit,
                                            'drift_fraction_limit': drift_fraction_limit,
                                            'drift_stderr_limit': drift_stderr_limit},
                              'conditions': {}, 'passed': False}
    fraction = float(abs(storage_rate/reference)) if reference is not None and storage_rate is not None \
        and np.isfinite(storage_rate) else None
    result['conditions']['storage'] = {'value': fraction,
                                      'passed': fraction is not None and fraction <= storage_fraction_limit,
                                      'reason': '' if fraction is not None else 'no usable positive injection/storage'}
    halves = half_window_comparison(times, values, start=start)
    result['halves'] = halves
    difference = halves.get('difference')
    drift = abs(difference)/reference if difference is not None and reference else None
    ratio = halves.get('difference_over_combined_stderr')
    constant_equal = halves['status'] == 'constant' and difference == 0
    ok_ratio = constant_equal or ratio is not None and abs(ratio) <= drift_stderr_limit
    result['conditions']['drift'] = {'stderr_ratio': ratio, 'fraction_over_injection': drift,
                                    'passed': bool(ok_ratio and drift is not None and drift <= drift_fraction_limit)}
    failed = [name for name, check in result['conditions'].items() if not check['passed']]
    result.update(passed=not failed, reason='failed: '+', '.join(failed) if failed else '')
    return result


def stationarity_screen(times: np.ndarray, flux: np.ndarray, *, injection: np.ndarray,
                        small: np.ndarray, large: np.ndarray, energy: np.ndarray,
                        start: float, block: float, t0: float = 1.) -> dict[str, Any]:
    """Two consecutive non-overlapping windows; no search for a passing burn-in.

    In each: storage <=2% injection, flux drift <=2 combined SE and <=5% injection,
    and injection/each dissipation's half-window drift <=5% its full-window mean.
    Both windows and the full-span half comparison must pass. Block/ESS checks
    are separate and required for ``statistics_qualified``. Time is in t0 units;
    energy/power stay in physical units. Constant Q=0 is not applicable, not a pass.
    """
    t, x, interval = _series(times, flux, start)
    auxiliary = {'injection': np.asarray(injection), 'small_dissipation': np.asarray(small),
                 'large_dissipation': np.asarray(large), 'energy': np.asarray(energy)}
    if not np.isfinite([block, t0]).all() or block <= 0 or t0 <= 0 or \
            not np.isclose(block/interval, round(block/interval), rtol=0, atol=1e-7):
        raise ValueError('Require positive t0 and a block aligned to the sampling interval')
    for data in auxiliary.values():
        _series(times, data, start)
    result: dict[str, Any] = {'protocol': 'two_nonoverlap_windows_v1', 'start': start,
                              'passed': False, 'statistics_qualified': False, 'windows': [],
                              'thresholds': {'storage_fraction': .02, 'flux_drift_over_injection': .05,
                                             'flux_drift_se': 2., 'power_dissipation_drift': .05}}
    if len(x) < 32:
        return {**result, 'reason': 'insufficient samples for two evaluation windows'}
    keep = np.asarray(times) > start+1e-10
    selected = {name: data[keep] for name, data in auxiliary.items()}
    if all(np.all(data == 0) for data in (x, *selected.values())):
        return {**result, 'applicable': False, 'reason': 'zero-field control: no injection or transport'}
    result['applicable'] = True
    split = len(x)//2
    bounds = ((0, split, start), (split, len(x), float(t[split-1])))
    for left, right, begin in bounds:
        stop = float(t[right-1])
        idx = (np.asarray(times) > begin+1e-10) & (np.asarray(times) <= stop+1e-10)
        wx = np.asarray(flux)[idx]
        p = np.asarray(injection)[idx]
        e_begin = float(np.interp(begin, times, energy))
        e_end = float(np.interp(stop, times, energy))
        storage = (e_end-e_begin)/((stop-begin)*t0)
        check = screen(np.asarray(times)[idx], wx, start=begin,
                       injection_reference=float(np.mean(p)), storage_rate=storage)
        conditions = check['conditions']
        for name in ('injection', 'small_dissipation', 'large_dissipation'):
            data = auxiliary[name][idx]
            mid = len(data)//2
            difference = abs(float(np.mean(data[mid:])-np.mean(data[:mid])))
            mean = float(np.mean(data))
            fraction = difference/abs(mean) if abs(mean) > 1e-30 else None
            conditions[name+'_drift'] = {'fraction': fraction,
                                         'passed': bool(fraction <= .05) if fraction is not None
                                         else bool(np.all(data == 0))}
        correlation = integrated_correlation_time(wx, interval)
        blocks = int(np.floor((stop-begin)/block+1e-9))
        quality = interval_quality(correlation, block, blocks)
        failed = [name for name, cond in conditions.items() if not cond['passed']]
        result['windows'].append({'start': begin, 'end': stop, 'passed': not failed,
                                  'reason': ', '.join(failed), 'conditions': conditions,
                                  'halves': check['halves'], 'correlation': correlation,
                                  'auxiliary_correlation': {
                                      name: integrated_correlation_time(data[idx], interval)
                                      for name, data in auxiliary.items()},
                                  'n_blocks': blocks, 'interval_quality': quality})
    e0 = float(np.interp(start, times, energy))
    total_storage = (float(np.interp(t[-1], times, energy))-e0)/((t[-1]-start)*t0)
    full = screen(times, flux, start=start, injection_reference=float(np.mean(selected['injection'])),
                  storage_rate=total_storage)
    result['full_window'] = full
    result['last_two_quarters'] = result['windows'][1]['halves']
    result['passed'] = full['passed'] and all(w['passed'] for w in result['windows'])
    result['statistics_qualified'] = result['passed'] and all(
        w['interval_quality']['qualified'] for w in result['windows'])
    result['reason'] = '' if result['passed'] else 'failed: full-span or consecutive-window screen'
    return result

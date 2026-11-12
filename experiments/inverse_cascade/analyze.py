"""Read explicit complete runs; plot fluxes and finite-window block uncertainty.

No integrator or physical model is constructed here. Burn-in and block duration
are analysis choices in t0 units, never silently optimized to obtain a desired sign.

Reported quantities keep two distinct normalizations apart, because they differ by
the storage term away from a statistical steady state:

* ``mean_Pi_*``             raw PDE flux through the cutoff (injection units);
* ``mean_Pi_*_over_injection``  divided by the window-mean injection of that field;
* ``large_*_fraction``      large-scale dissipation / total dissipation of the field;
* ``large_*_over_injection`` large-scale dissipation / mean injection.

A zero or unavailable denominator yields ``None`` instead of a silent division by one.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import t as student_t

from postprocess.plotting import latex_style
from experiments.records import PROJECT, provenance, write_json
from . import stats
from .records import load

SCHEMA_VERSION = 3
HOMOGENEOUS_KEYS = ('nx', 'ny', 'kf', 'f0', 'scheme', 'gamma', 'noise_rms', 'dt', 'T',
                    'Re_plus', 'Re_minus', 'sample_every')


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Normalized value, or None when the denominator cannot support a ratio."""
    if numerator is None or denominator is None:
        return None
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 1e-30:
        return None
    return float(numerator/denominator)


def _interval_quality(correlation: dict[str, Any], block: float, n_blocks: int) -> dict[str, Any]:
    """Pre-registered requirements for treating a block interval as usable."""
    return stats.interval_quality(correlation, block, n_blocks)


def _denominator_note(value: float | None) -> str:
    if value is None:
        return 'denominator unavailable'
    if not np.isfinite(value):
        return 'denominator not finite'
    if value <= 1e-30:
        return 'denominator is zero or nonpositive: no ratio reported'
    return ''


def block_statistics(times: np.ndarray, values: np.ndarray, *, start: float,
                     duration: float) -> dict[str, Any]:
    """Equal-time samples; complete disjoint (left,right] blocks; Student-t heuristic.

    The plain ``mean``/``std`` use every post-burn-in sample. ``block_means`` holds
    only complete blocks, so the interval never mixes a full block with a truncated
    tail; ``tail_dropped_t0`` records how much of the window the interval ignores.
    These intervals require approximately stationary, independent block means.
    Neither condition follows merely from running this function.
    """
    if not np.isfinite([start, duration]).all() or start < 0 or duration <= 0:
        raise ValueError('Require nonnegative burn-in and positive block duration')
    if times.ndim != 1 or values.shape != times.shape or len(times) < 2 \
            or not np.isfinite(times).all() or not np.isfinite(values).all() \
            or np.any(np.diff(times) <= 0) \
            or not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-7, atol=1e-10):
        raise ValueError('Require finite uniformly sampled scalar series')
    interval = float(times[1]-times[0])
    if not np.isclose(duration/interval, round(duration/interval), rtol=0, atol=1e-7):
        raise ValueError('Block duration must be a multiple of the flux sampling interval')
    selected = values[times > start+1e-10]
    if len(selected) < 2:
        raise ValueError('Insufficient samples after burn-in')
    means, endpoints = [], []
    n = int(np.floor((times[-1]-start)/duration+1e-9))
    for i in range(n):
        left, right = start+i*duration, start+(i+1)*duration
        chunk = values[(times > left+1e-10) & (times <= right+1e-10)]
        if len(chunk) < 2:
            raise ValueError('Each block needs at least two flux samples')
        means.append(float(np.mean(chunk)))
        endpoints.append(right)
    prefix = []
    for i in range(1, len(means)+1):
        half = (float(student_t.ppf(.975, i-1)*np.std(means[:i], ddof=1)/np.sqrt(i))
                if i >= 4 else None)
        prefix.append({'end': endpoints[i-1], 'mean': float(np.mean(means[:i])), 'ci95_halfwidth': half})
    mean, std = float(np.mean(selected)), float(np.std(selected, ddof=1))
    correlation = stats.integrated_correlation_time(selected, interval)
    stderr = (float(std/np.sqrt(correlation['n_eff']))
              if correlation['status'] == 'ok' and correlation['n_eff'] else None)
    quality = _interval_quality(correlation, duration, len(means))
    return {'mean': mean, 'std': std,
            'ci95_mean': prefix[-1]['mean'] if prefix else None,
            'ci95_halfwidth': prefix[-1]['ci95_halfwidth'] if prefix else None,
            'ci95_center_definition': 'mean of complete blocks only; excludes tail',
            'positive_fraction': float(np.mean(selected > 0)),
            'negative_fraction': float(np.mean(selected < 0)),
            'std_over_abs_mean': std/abs(mean) if abs(mean) > 1e-14*max(1., std) else None,
            'block_means': means, 'prefix': prefix,
            'mean_complete_blocks': float(np.mean(means)) if means else None,
            'n_blocks': len(means), 'block_t0': float(duration),
            'tail_dropped_t0': float(max(0., times[-1]-(start+n*duration))),
            'sample_interval': interval, 'n_samples': int(len(selected)),
            'stderr': stderr, 'correlation': correlation, 'interval_quality': quality,
            'ci_caveat': 'Heuristic: stationarity and independence of blocks not established.'}


def _budget(d: Any, c: dict, start: float, quantity: str) -> dict[str, Any]:
    times = d['times'][:]/c['t0']
    physical = d['times'][:]
    keep = times >= start-1e-10
    ts = np.r_[start*c['t0'], physical[keep & (times > start+1e-10)]]
    field = 'total_energy' if quantity == 'E' else 'potential_squared'
    energy = np.interp(ts, physical, d['diagnostics/'+field][:])
    power_key = 'power' if quantity == 'E' else 'power_Q'
    p = np.interp(ts, physical, d['diagnostics/'+power_key][:])
    small = np.interp(ts, physical, d[f'diagnostics/dissipation_{quantity}_small'][:])
    large = np.interp(ts, physical, d[f'diagnostics/dissipation_{quantity}_large'][:])
    span = ts[-1]-ts[0]
    power, ds, dl = [float(trapezoid(v, ts)/span) for v in (p, small, large)]
    change = float(energy[-1]-energy[0])
    storage = change/span
    return {'notes': {'large_fraction_of_dissipation': _denominator_note(ds+dl),
                      'large_over_injection': _denominator_note(power),
                      'storage_over_injection': _denominator_note(power)},
            'mean_injection': power, 'mean_small_dissipation': ds, 'mean_large_dissipation': dl,
            'large_fraction_of_dissipation': _ratio(dl, ds+dl),
            'large_over_injection': _ratio(dl, power),
            'mean_storage_rate': storage,
            'storage_over_injection': _ratio(storage, power),
            'integrated_budget_defect': change-float(trapezoid(p-small-large, ts))}


LOWPASS_QUALIFICATION = 1e-3


def _lowpass_budget(d: h5py.File, c: dict, start: float, cutoff: int,
                    threshold: float = LOWPASS_QUALIFICATION) -> dict[str, Any]:
    """Check ``dX_<K/dt = P_<K - D+_<K - D-_<K - Pi_X(K)`` for both fields.

    Uses the exact shell-summed low-pass columns; no shell-centre approximation.
    The instantaneous rate is a finite difference of the sampled low-pass energy, so
    the defect carries that discretisation error and is reported, not hidden.
    """
    if 'spectra/E_low' not in d:
        return {'status': 'unavailable', 'reason': 'record has no low-pass budget columns'}
    ts = d['spectra/times'][:]/c['t0']
    physical = d['spectra/times'][:]
    keep = ts > start+1e-10
    if np.count_nonzero(keep) < 8:
        return {'status': 'unavailable', 'reason': 'too few samples after burn-in'}
    times = physical[keep]
    out: dict[str, Any] = {'status': 'ok', 'K': int(cutoff), 'sample_interval': float(times[1]-times[0]),
                          'normalization': 'total injection of that field (K=max), not the low-pass injection',
                          'note': ('window defect is the qualification quantity; the instantaneous maximum is '
                                   'limited by the finite difference of the sampled low-pass energy')}
    for field, energy, power, plus, minus, flux in (
            ('E', 'E_low', 'power_E_low', 'diss_E_plus_low', 'diss_E_minus_low', 'Pi_E'),
            ('Q', 'Q_low', 'power_Q_low', 'diss_Q_plus_low', 'diss_Q_minus_low', 'Pi_Q')):
        low = d['spectra/'+energy][:][keep]
        injected = d['spectra/'+power][:][keep]
        dplus = d['spectra/'+plus][:][keep]
        dminus = d['spectra/'+minus][:][keep]
        transferred = d['spectra/'+flux][:][keep]
        residual = injected-dplus-dminus-transferred
        defect = np.gradient(low, times, axis=0)-residual
        # Low-pass injection vanishes below forcing; use the full-field mean injection.
        span = float(times[-1]-times[0])
        scale = float(trapezoid(injected[:, -1], times)/span)
        window_defect = (low[-1]-low[0])/span-trapezoid(residual, times, axis=0)/span
        ratios = np.abs(window_defect)/scale if scale > 0 else None
        maximum = float(np.max(ratios)) if ratios is not None else None
        selected_max = float(np.max(np.abs(defect[:, cutoff])))
        out[field] = {
            'max_abs_defect': selected_max,
            'max_abs_defect_over_total_injection': selected_max/scale if scale > 0 else None,
            'window_defect_over_total_injection': float(ratios[cutoff]) if ratios is not None else None,
            'window_defect_over_total_injection_all_K': (ratios.tolist() if ratios is not None
                                                        else [None]*low.shape[1]),
            'max_abs_window_defect_over_total_injection': maximum,
            'threshold': float(threshold),
            'passed': maximum is not None and maximum <= threshold,
            'total_injection': scale,
            'mean_injection_low': float(trapezoid(injected[:, cutoff], times)/span),
            'mean_flux_low': float(trapezoid(transferred[:, cutoff], times)/span)}
    return out


def _homogeneous(sources: list[dict[str, Any]]) -> str:
    """Why a shared mu series would be rejected, or an empty string when it is valid."""
    configs = [source['c'] for source in sources]
    mus = [c['mu'] for c in configs]
    if len(set(mus)) != len(mus):
        return 'duplicate mu in comparison group'
    for key in HOMOGENEOUS_KEYS:
        values = {c[key] for c in configs}
        if len(values) != 1:
            return f'mixed {key} in comparison group: {sorted(values)}'
    return ''


def run_source(path: Path) -> dict[str, Any]:
    """One complete single-run record, opened lazily so analysis stays read-only."""
    c, manifest = load(path, require_complete=True)
    return {'name': str(path.resolve()), 'c': c, 'manifest': manifest,
            'open': lambda: h5py.File(path/'results.h5', 'r')}


def analyze(paths: list[Path], output: Path, *, burn_in: float, block: float,
            K: int | None = None) -> dict[str, Any]:
    if not paths:
        raise ValueError('Provide explicit runs')
    return analyze_sources([run_source(path) for path in paths], output,
                           burn_in=burn_in, block=block, K=K)


def analyze_sources(sources: list[dict[str, Any]], output: Path, *, burn_in: float,
                    block: float, K: int | None = None) -> dict[str, Any]:
    """Analyze single runs or segmented chains through one statistics path."""
    if not sources:
        raise ValueError('Provide explicit runs')
    if not np.isfinite([burn_in, block]).all() or burn_in < 0 or block <= 0:
        raise ValueError('Invalid analysis window')
    if any(burn_in >= source['c']['T'] for source in sources):
        raise ValueError('Burn-in must precede every end time')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new report directory; historical reports are not overwritten')
    output.mkdir(parents=True, exist_ok=True)
    rc = {'figure.dpi': 160, 'savefig.dpi': 200}
    with latex_style(rc):
        return _plots(sources, output, burn_in=burn_in, block=block, K=K)


def _plots(sources: list[dict[str, Any]], output: Path, *, burn_in: float, block: float,
           K: int | None) -> dict[str, Any]:
    fluxfig, fluxaxes = plt.subplots(2, 1, figsize=(7.2, 6.4), layout='constrained')
    rawfig, rawaxes = plt.subplots(2, 1, figsize=(7.2, 6.4), layout='constrained')
    seriesfig, seriesaxes = plt.subplots(len(sources), 1, figsize=(9, 2.2*len(sources)),
                                         squeeze=False, layout='constrained')
    windowfig, windowaxes = plt.subplots(2, 1, figsize=(7.2, 6.4), layout='constrained')
    specfig, specaxes = plt.subplots(2, 1, figsize=(7.2, 6.4), layout='constrained')
    summary: dict[str, Any] = {
        'analysis_schema_version': SCHEMA_VERSION,
        'burn_in_t0': burn_in, 'block_t0': block, 'analysis_source': provenance(),
        'normalization': {'mean_Pi_*_over_injection': 'divided by the window-mean injection',
                          'large_*_fraction': 'large-scale dissipation / total dissipation',
                          'large_*_over_injection': 'large-scale dissipation / mean injection',
                          'null': 'denominator nonpositive, non-finite or unavailable'},
        'limits': ('Exploratory finite-window statistics. Not a critical-value, exponent, '
                   'stationarity or convergence qualification.'),
        'runs': []}
    rows = []
    for index, source in enumerate(sources):
        c, manifest = source['c'], source['manifest']
        cutoff = K if K is not None else max(1, int(np.floor(.75*np.sqrt(2)*c['kf'])))
        if type(cutoff) is not int or cutoff <= 0 or cutoff >= np.sqrt(2)*c['kf']:
            raise ValueError('Representative K must be positive and below radial forcing sqrt(2)*kf')
        label = (f'$\\mu={c["mu"]:g}$, {c["scheme"]}, $N={c["nx"]}$, '
                 f'$\\Delta t^*={c["dt"]:g}$, $T^*={c["T"]:g}$, seed {c["seed"]}, '
                 f'window ({burn_in:g},{c["T"]:g}], block {block:g}')
        with source['open']() as d:
            ts = d['spectra/times'][:]/c['t0']
            ks = d['spectra/K'][:]
            keep = ts > burn_in+1e-10
            stats_by_field = {q: block_statistics(ts, d['spectra/Pi_'+q][:, cutoff],
                                                 start=burn_in, duration=block)
                              for q in ('E', 'Q')}
            budgets = {q: _budget(d, c, burn_in, q) for q in ('E', 'Q')}
            lowpass = _lowpass_budget(d, c, burn_in, cutoff)
            for q, ax in zip(('E', 'Q'), fluxaxes):
                mean_flux = np.mean(d['spectra/Pi_'+q][:][keep], axis=0)
                injection = budgets[q]['mean_injection']
                normalized = _ratio(mean_flux[cutoff], injection) is not None
                symbol = 'E' if q == 'E' else 'A'
                if normalized:
                    ax.plot(ks/c['kf'], mean_flux/injection, label=label)
                else:
                    ax.plot([], [], label=label+' (normalization unavailable)')
                ax.set(xlabel='$K/k_f$', ylabel=f'$\\Pi_{symbol}(K)\\,/\\,\\langle P_{symbol}\\rangle$')
                rawax = rawaxes[0 if q == 'E' else 1]
                rawax.plot(ks/c['kf'], mean_flux, label=label)
                rawax.set(xlabel='$K/k_f$', ylabel=f'$\\Pi_{symbol}(K)$ (raw)')
                rawax.axhline(0, color='0.5', lw=.6)
                rawax.axvline(np.sqrt(2), color='0.6', ls=':', lw=.8)
                ax.axhline(0, color='0.5', lw=.6)
                ax.axvline(np.sqrt(2), color='0.6', ls=':', lw=.8,
                           label=r'$\sqrt{2}k_f$ radial forcing shell')
                ax.grid(alpha=.25)
            if budgets['Q']['mean_injection'] is not None and \
                    abs(budgets['Q']['mean_injection']) <= 1e-30:
                fluxaxes[1].text(.02, .04, 'zero injection: see mean_flux_raw for $\\Pi_A$',
                                 transform=fluxaxes[1].transAxes, fontsize=7, color='0.35')
            flux_series = d['spectra/Pi_E'][:, cutoff]
            ax = seriesaxes[index, 0]
            ax.plot(ts, flux_series, lw=.6)
            ax.axhline(stats_by_field['E']['mean'], color='C1', label='post-burn mean')
            ax.axvline(burn_in, color='0.5', ls=':')
            ax.set(title=label+f', $K={cutoff}$', xlabel='$t/t_0$', ylabel='$\\Pi_E(t,K)$')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            for q, ax in zip(('E', 'Q'), windowaxes):
                symbol = 'E' if q == 'E' else 'A'
                prefix = stats_by_field[q]['prefix']
                if prefix:
                    xx = [v['end']-burn_in for v in prefix]
                    yy = np.array([v['mean'] for v in prefix])
                    hh = np.array([v['ci95_halfwidth'] if v['ci95_halfwidth'] is not None else np.nan
                                   for v in prefix])
                    line, = ax.plot(xx, yy, ls=('-', '--', ':', '-.')[index % 4], label=label)
                    ax.fill_between(xx, yy-hh, yy+hh, color=line.get_color(), alpha=.12)
                ax.set(xlabel='observation window$\\,/\\,t_0$',
                       ylabel=f'block mean $\\Pi_{symbol}(K)$, 95\\% heuristic interval')
                ax.axhline(0, color='0.5', lw=.6)
                ax.grid(alpha=.25)
            for key, ax in zip(('E_u', 'E_B'), specaxes):
                values = np.mean(d['spectra/'+key][:][keep], axis=0)
                positive = values[1:] > 0
                ax.loglog((ks[1:]/c['kf'])[positive], values[1:][positive], label=label)
                ax.set(xlabel='$K/k_f$', ylabel=('$E_u(K)$' if key == 'E_u' else '$E_b(K)$')+', shell sum')
                ax.grid(alpha=.25, which='both')
            coupling = (d['corrections/frozen_factor'][:]*(1-d['stages/r'][:])
                        if c['scheme'].endswith('mrsav') else np.ones_like(d['stages/r'][:]))
            stage_window = d['times'][1:]/c['t0'] > burn_in+1e-10
            errors = {key: float(np.max(d['stages/'+key][:])) for key in
                      ('residual', 'scalar_residual', 'divergence_inf', 'continuity_residual')}
            diagnostic_times = d['times'][:]/c['t0']
            screens = {}
            for q in ('E', 'Q'):
                def sampled(key: str) -> np.ndarray:
                    return np.interp(ts, diagnostic_times, d['diagnostics/'+key][:])
                screens[q] = stats.stationarity_screen(
                    ts, d['spectra/Pi_'+q][:, cutoff], start=burn_in, block=block, t0=c['t0'],
                    injection=sampled('power' if q == 'E' else 'power_Q'),
                    small=sampled(f'dissipation_{q}_small'), large=sampled(f'dissipation_{q}_large'),
                    energy=sampled('total_energy' if q == 'E' else 'potential_squared'))
            record = {'run': source['name'], 'results_sha256': manifest['results_sha256'],
                      'config_sha256': manifest['config_sha256'], 'mu': c['mu'], 'scheme': c['scheme'],
                      'nx': c['nx'], 'dt_t0': c['dt'], 'T_t0': c['T'], 'seed': c['seed'],
                      'Re_plus': c['Re_plus'], 'Re_minus': c['Re_minus'],
                      'K': cutoff, 'statistics': stats_by_field, 'budgets': budgets,
                      'stationarity': screens, 'lowpass_budget': lowpass,
                      'max_abs_coupling_minus_one': float(np.max(np.abs(coupling-1))),
                      'window_max_abs_coupling_minus_one': (
                          float(np.max(np.abs(coupling[stage_window]-1))) if stage_window.any() else None),
                      'max_abs_r': float(np.max(np.abs(d['diagnostics/r'][:]))),
                      'max_stage_errors': errors}
            summary['runs'].append(record)
            normalized_e = _ratio(stats_by_field['E']['mean'], budgets['E']['mean_injection'])
            half_e = stats_by_field['E']['prefix'][-1]['ci95_halfwidth'] if stats_by_field['E']['prefix'] else None
            rows.append({
                'run': Path(source['name']).name, 'mu': c['mu'], 'scheme': c['scheme'], 'nx': c['nx'],
                'dt_t0': c['dt'], 'T_t0': c['T'], 'seed': c['seed'],
                'K': cutoff,
                'mean_Pi_E': stats_by_field['E']['mean'],
                'std_Pi_E': stats_by_field['E']['std'],
                'negative_fraction': stats_by_field['E']['negative_fraction'],
                'ci_Pi_E': half_e,
                'block_mean_Pi_E': stats_by_field['E']['ci95_mean'],
                'block_mean_Pi_E_over_injection': _ratio(stats_by_field['E']['ci95_mean'], budgets['E']['mean_injection']),
                'mean_Pi_E_over_injection': normalized_e,
                'ci_Pi_E_over_injection': _ratio(half_e, budgets['E']['mean_injection']),
                'mean_Pi_Q': stats_by_field['Q']['mean'],
                'mean_Pi_Q_over_injection': _ratio(stats_by_field['Q']['mean'], budgets['Q']['mean_injection']),
                'ci_Pi_Q_over_injection': _ratio(stats_by_field['Q']['ci95_halfwidth'], budgets['Q']['mean_injection']),
                'block_mean_Pi_Q': stats_by_field['Q']['ci95_mean'],
                'std_Pi_Q': stats_by_field['Q']['std'],
                'ci_Pi_Q': stats_by_field['Q']['prefix'][-1]['ci95_halfwidth']
                          if stats_by_field['Q']['prefix'] else None,
                'large_E_fraction': budgets['E']['large_fraction_of_dissipation'],
                'large_Q_fraction': budgets['Q']['large_fraction_of_dissipation'],
                'large_E_over_injection': budgets['E']['large_over_injection'],
                'large_Q_over_injection': budgets['Q']['large_over_injection'],
                'E_storage_over_injection': budgets['E']['storage_over_injection'],
                'sigma_E_over_large_dissipation': _ratio(
                    stats_by_field['E']['std'], budgets['E']['mean_large_dissipation']),
                'sigma_Q_over_large_dissipation': _ratio(
                    stats_by_field['Q']['std'], budgets['Q']['mean_large_dissipation']),
                'tau_int_Pi_E': stats_by_field['E']['correlation'].get('tau_int'),
                'n_eff_Pi_E': stats_by_field['E']['correlation'].get('n_eff'),
                'stderr_Pi_E': stats_by_field['E']['stderr'],
                'n_blocks': stats_by_field['E']['n_blocks'],
                'interval_qualified': stats_by_field['E']['interval_quality']['qualified'],
                'block_over_tau_int': stats_by_field['E']['interval_quality']['block_over_tau_int'],
                'tail_dropped_t0': stats_by_field['E']['tail_dropped_t0'],
                'lowpass_defect_E': (lowpass.get('E') or {}).get('max_abs_defect_over_total_injection')
                                    if lowpass['status'] == 'ok' else None,
                'lowpass_window_defect_E': (lowpass.get('E') or {}).get('window_defect_over_total_injection')
                                           if lowpass['status'] == 'ok' else None,
                'lowpass_passed_E': (lowpass.get('E') or {}).get('passed')
                                    if lowpass['status'] == 'ok' else None,
                'stationarity_E': screens['E']['passed'],
                'stationarity_E_reason': screens['E']['reason'],
                'max_coupling_change': record['max_abs_coupling_minus_one']})
    rejection = _homogeneous(sources)
    summary['comparison_group'] = {'homogeneous': rejection == '', 'reason': rejection}
    if len(rows) >= 3 and not rejection:
        _transition_figure(rows, output)
    elif len(rows) >= 3:
        summary['transition_skipped'] = rejection
    for figure, axes, name in ((fluxfig, fluxaxes, 'mean_flux'),
                               (rawfig, rawaxes, 'mean_flux_raw'),
                               (windowfig, windowaxes, 'window_statistics'),
                               (specfig, specaxes, 'spectra')):
        for ax in axes:
            ax.legend(fontsize=6)
        for suffix in ('png', 'pdf'):
            figure.savefig(output/(name+'.'+suffix), dpi=160)
        plt.close(figure)
    for suffix in ('png', 'pdf'):
        seriesfig.savefig(output/('instantaneous_flux.'+suffix), dpi=160)
    plt.close(seriesfig)
    with (output/'statistics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary['statistics_table'] = rows
    write_json(output/'summary.json', summary)
    return summary


def _transition_figure(rows: list[dict[str, Any]], output: Path) -> None:
    """Paper Fig. 1/3 analogue over one homogeneous comparison group."""
    order = np.argsort([row['mu'] for row in rows])
    mu = np.asarray([row['mu'] for row in rows])[order]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), layout='constrained')
    axes = axes.ravel()
    normalized = np.asarray([row['block_mean_Pi_E_over_injection']
                             if row['block_mean_Pi_E_over_injection'] is not None else np.nan
                             for row in rows], dtype=float)[order]
    half = np.asarray([row['ci_Pi_E_over_injection'] if row['ci_Pi_E_over_injection'] is not None
                       else np.nan for row in rows], dtype=float)[order]
    raw = np.asarray([row['block_mean_Pi_E'] if row['block_mean_Pi_E'] is not None
                      else np.nan for row in rows])[order]
    ax = axes[0]
    if np.isfinite(normalized).all():
        ax.errorbar(mu, normalized, yerr=half, fmt='o-', capsize=3)
        ax.set(ylabel=r'$\langle\Pi_E(K)\rangle/\langle P_E\rangle$')
    else:
        ax.plot(mu, raw, 'o-')
        ax.set(ylabel=r'$\langle\Pi_E(K)\rangle$ (raw: injection unavailable)')
    ax.axhline(0, color='0.6', lw=.6)
    ax.set(xlabel=r'$\mu$')
    ax.grid(alpha=.3)
    for ax, key, label in zip(axes[1:3], ('large_E_fraction', 'large_Q_fraction'),
                              (r'$D^{-}_{E}/(D^{+}_{E}+D^{-}_{E})$',
                               r'$D^{-}_{A}/(D^{+}_{A}+D^{-}_{A})$')):
        values = np.asarray([row[key] if row[key] is not None else np.nan for row in rows],
                            dtype=float)[order]
        ax.plot(mu, values, 'o-')
        ax.axhline(0, color='0.6', lw=.6)
        ax.set(xlabel=r'$\mu$', ylabel=label)
        ax.grid(alpha=.3)
    ax = axes[3]
    for key, label in (('sigma_E_over_large_dissipation', r'$\sigma_E/\epsilon^{-}_{E}$'),
                       ('sigma_Q_over_large_dissipation', r'$\sigma_A/\epsilon^{-}_{A}$')):
        values = np.asarray([row[key] if row[key] is not None else np.nan for row in rows],
                            dtype=float)[order]
        ax.semilogy(mu, values, 'o-', label=label)
    ax.set(xlabel=r'$\mu$', ylabel=r'$\sigma/\epsilon^{-}$')
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8)
    ax.set_title('finite window: no divergence claim', fontsize=8)
    figure.suptitle('cascade transition vs magnetic forcing ratio $\\mu$ '
                    '(one homogeneous group, finite window)', fontsize=10)
    for suffix in ('png', 'pdf'):
        figure.savefig(output/('transition.'+suffix), dpi=160)
    plt.close(figure)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='*')
    p.add_argument('--batch', type=Path)
    p.add_argument('--chain', type=Path, action='append', default=None)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--burn-in', type=float, required=True)
    p.add_argument('--block', type=float, required=True)
    p.add_argument('--K', type=int)
    a = p.parse_args()
    paths = list(a.runs)
    sources = sum(v is not None for v in (a.batch, a.chain))
    if sources > 1 or (sources and paths):
        p.error('Provide runs, --batch or --chain, and only one of them')
    if a.batch:
        batch = json.loads(a.batch.read_text())
        if batch['status'] != 'complete':
            raise ValueError('Complete scan required for statistical comparison')
        paths = [Path(v['run']) for v in batch['members']]
    if a.chain:
        from .chain import chain_source
        analyze_sources([chain_source(v) for v in a.chain], a.output,
                        burn_in=a.burn_in, block=a.block, K=a.K)
        print(a.output)
        return
    paths = [v if v.is_dir() else PROJECT/'runs'/'inverse_cascade'/v for v in paths]
    analyze(paths, a.output, burn_in=a.burn_in, block=a.block, K=a.K)
    print(a.output)


if __name__ == '__main__':
    main()

"""Statistics, normalization and low-pass budget mechanisms of the cascade study."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.inverse_cascade import stats
from experiments.inverse_cascade.analyze import (LOWPASS_QUALIFICATION, _lowpass_budget,
                                                analyze, analyze_sources, block_statistics)
from experiments.inverse_cascade.chain import chain_source, load_chain
from experiments.inverse_cascade.plan import BASE, budget, matrix
from experiments.inverse_cascade.records import load
from experiments.inverse_cascade.run import run, run_scan
from experiments.inverse_cascade.segments import execution_settings, run_segmented

SMALL = {'nx': 24, 'ny': 24, 'kf': 2, 'T': .4, 'dt': .01, 'sample_every': 2, 'mu': .3,
         'gamma': 1000.}


def _ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi*x[i-1]+rng.standard_normal()
    return x


def test_integrated_correlation_time_known_cases():
    interval = .1
    times = np.arange(4001)*interval
    # White noise: tau_int -> interval/2, n_eff -> number of samples.
    white = np.random.default_rng(3).standard_normal(len(times))
    estimate = stats.integrated_correlation_time(white, interval)
    assert estimate['status'] == 'ok'
    assert .5 < estimate['tau_int']/interval < 1.5
    # Strongly correlated series must report a large tau_int and few effective samples.
    slow = _ar1(len(times), .95)
    slow_estimate = stats.integrated_correlation_time(slow, interval)
    assert slow_estimate['status'] == 'ok'
    assert slow_estimate['tau_int'] > 5*interval
    assert slow_estimate['n_eff'] < estimate['n_eff']
    # Constant and too-short series must not fabricate a finite error.
    assert stats.integrated_correlation_time(np.full(200, 3.), interval)['status'] == 'constant'
    assert stats.integrated_correlation_time(np.arange(4.), interval)['status'] == 'insufficient'


def test_half_window_screen_flags_drift_but_not_noise():
    interval = .1
    rng = np.random.default_rng(5)
    times = np.arange(2000)*interval
    steady = 1.+0.1*rng.standard_normal(len(times))
    drifted = 1.+0.3*times/times[-1]+0.01*rng.standard_normal(len(times))
    good = stats.screen(times, steady, start=0., injection_reference=1., storage_rate=1e-4)
    bad = stats.screen(times, drifted, start=0., injection_reference=1., storage_rate=1e-4)
    assert good['passed'] is True
    assert bad['passed'] is False
    assert bad['conditions']['drift']['passed'] is False
    storage = stats.screen(times, steady, start=0., injection_reference=1., storage_rate=.5)
    assert storage['conditions']['storage']['passed'] is False
    # No injection reference means the storage condition cannot pass silently.
    unavailable = stats.screen(times, steady, start=0., injection_reference=None, storage_rate=None)
    assert unavailable['passed'] is False


def test_block_statistics_reports_complete_blocks_and_tail():
    times = np.arange(0., 21., 1.)
    values = np.linspace(1., 2., len(times))
    # 20 time units of data, blocks of 7 leave a 6-unit tail that must be reported.
    stats_out = block_statistics(times, values, start=0., duration=7.)
    assert stats_out['n_blocks'] == 2
    assert len(stats_out['block_means']) == 2
    assert stats_out['tail_dropped_t0'] == pytest.approx(6.)
    # ``mean`` uses samples strictly after the burn-in boundary, as documented.
    assert stats_out['mean'] == pytest.approx(float(np.mean(values[1:])))
    assert stats_out['mean_complete_blocks'] == pytest.approx(float(np.mean(stats_out['block_means'])))
    empty = block_statistics(times, values, start=0., duration=25.)
    assert empty['n_blocks'] == 0


def test_normalization_fields_and_null_denominator(tmp_path):
    magnetic = run({**SMALL, 'mu': .5}, root=tmp_path)
    control = run({**SMALL, 'mu': 0.}, root=tmp_path)
    summary = analyze([magnetic, control], tmp_path/'report', burn_in=.1, block=.06, K=2)
    assert summary['analysis_schema_version'] == 3
    forced, unforced = summary['runs']
    injection = forced['budgets']['E']['mean_injection']
    assert forced['statistics']['E']['mean'] == pytest.approx(
        forced['statistics']['E']['mean'])
    rows = (tmp_path/'report'/'statistics.csv').read_text().splitlines()
    header = rows[0].split(',')
    assert 'mean_Pi_E_over_injection' in header
    assert 'large_E_over_injection' in header
    assert 'tau_int_Pi_E' in header
    # The HD control has no magnetic injection, so the A-channel ratio is null, not 1.
    assert unforced['budgets']['Q']['mean_injection'] == 0
    assert unforced['budgets']['Q']['large_over_injection'] is None
    assert forced['budgets']['Q']['mean_injection'] > 0
    assert abs(float(injection)) > 0
    # Both dissipation ratios are reported and must not be silently interchanged.
    assert forced['budgets']['E']['large_fraction_of_dissipation'] is not None
    assert forced['budgets']['E']['large_over_injection'] is not None


def test_transition_figure_requires_a_homogeneous_group(tmp_path):
    a = run({**SMALL, 'mu': .2}, root=tmp_path)
    b = run({**SMALL, 'mu': .4}, root=tmp_path)
    c = run({**SMALL, 'mu': .6, 'nx': 32, 'ny': 32}, root=tmp_path)
    summary = analyze([a, b, c], tmp_path/'mixed', burn_in=.1, block=.06, K=2)
    assert summary['comparison_group']['homogeneous'] is False
    assert 'mixed nx' in summary['comparison_group']['reason']
    assert not (tmp_path/'mixed'/'transition.png').exists()
    assert 'transition_skipped' in summary
    homogeneous = analyze([a, b], tmp_path/'pair', burn_in=.1, block=.06, K=2)
    assert homogeneous['comparison_group']['homogeneous'] is True


def test_lowpass_budget_columns_are_exact(tmp_path):
    path = run({**SMALL, 'mu': .3}, root=tmp_path)
    c, _ = load(path, require_complete=True)
    with h5py.File(path/'results.h5') as d:
        last = len(d['spectra/K'][:])-1
        # Global consistency of the new columns with the recorded diagnostics.
        np.testing.assert_allclose(d['spectra/E_low'][:][:, last], d['spectra/E_u'][:].sum(axis=1)
                                   + d['spectra/E_B'][:].sum(axis=1), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(d['spectra/power_E_low'][:][:, last],
                                   d['spectra/power_E_low'][:][:, last])
        np.testing.assert_allclose(
            d['spectra/diss_E_plus_low'][:][:, last]+d['spectra/diss_E_minus_low'][:][:, last],
            d['diagnostics/dissipation_E_small'][::c['sample_every']]
            + d['diagnostics/dissipation_E_large'][::c['sample_every']], rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(d['spectra/power_E_low'][:][:, last],
                                   d['diagnostics/power'][::c['sample_every']], rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(d['spectra/Q_low'][:][:, last],
                                   d['diagnostics/potential_squared'][::c['sample_every']],
                                   rtol=1e-12, atol=1e-16)
        # Below the forcing shell no energy is injected, and the flux identity closes.
        budget_below = _lowpass_budget(d, c, .1, 1)
        assert budget_below['status'] == 'ok'
        assert budget_below['E']['mean_injection_low'] < 1e-12
        assert budget_below['E']['window_defect_over_total_injection'] < LOWPASS_QUALIFICATION
        # Above the forcing shell the identity still closes within the window tolerance.
        budget_above = _lowpass_budget(d, c, .1, 3)
        assert budget_above['E']['window_defect_over_total_injection'] < LOWPASS_QUALIFICATION
        assert budget_above['E']['passed'] is True


def test_segmented_chain_matches_continuous_run_and_resumes(tmp_path):
    whole = run(SMALL, root=tmp_path)
    chain = run_segmented(SMALL, execution={'segment_steps': 10}, root=tmp_path)
    loaded = load_chain(chain, require_complete=True)
    c, _ = load(whole, require_complete=True)
    with h5py.File(whole/'results.h5') as d:
        np.testing.assert_array_equal(loaded['arrays']['times'], d['times'][:])
        np.testing.assert_allclose(loaded['arrays']['diagnostics']['total_energy'],
                                   d['diagnostics/total_energy'][:], rtol=0, atol=1e-15)
        np.testing.assert_allclose(loaded['arrays']['spectra']['Pi_E'],
                                   d['spectra/Pi_E'][:], rtol=0, atol=1e-15)
        np.testing.assert_allclose(loaded['arrays']['stages']['residual'],
                                   d['stages/residual'][:], rtol=0, atol=1e-14)
        reference = d['final/magnetic'][:]
    with h5py.File(chain/'segments'/'0003.h5') as segment:
        np.testing.assert_allclose(segment['final/magnetic'][:], reference, rtol=0, atol=1e-14)
    # Resume from a deliberately interrupted chain and reach the same terminal state.
    partial = run_segmented(SMALL, execution={'segment_steps': 10, 'stop_after_segment': 2},
                            root=tmp_path)
    interrupted = json.loads((partial/'chain.json').read_text())
    assert interrupted['status'] == 'running' and len(interrupted['segments']) == 2
    with pytest.raises(ValueError, match='has not reached'):
        load_chain(partial, require_complete=True)
    resumed = run_segmented(SMALL, execution={'segment_steps': 10, 'resume_from': partial},
                            root=tmp_path)
    assert load_chain(resumed, require_complete=True)['chain']['status'] == 'complete'
    summary = analyze_sources([chain_source(resumed)], tmp_path/'chainreport',
                              burn_in=.1, block=.06, K=2)
    assert summary['runs'][0]['statistics']['E']['n_blocks'] >= 1


def test_chain_rejects_tampering_and_bad_execution_settings(tmp_path):
    chain = run_segmented(SMALL, execution={'segment_steps': 10}, root=tmp_path)
    segment = chain/'segments'/'0001.h5'
    with h5py.File(segment, 'a') as d:
        d['times'][0] += 1e-9
    with pytest.raises(ValueError, match='checksum'):
        load_chain(chain, require_complete=True)
    # Execution settings never enter the physical identity and are validated strictly.
    for bad in ({'segment_steps': 0}, {'segment_steps': 1.5}, {'stop_after_segment': -1},
                {'unexpected': 1}):
        with pytest.raises(ValueError):
            execution_settings(bad, 100)
    # Resuming a complete chain or a different configuration is refused.
    fresh = run_segmented(SMALL, execution={'segment_steps': 10}, root=tmp_path)
    with pytest.raises(ValueError, match='already reached'):
        run_segmented(SMALL, execution={'segment_steps': 10, 'resume_from': fresh}, root=tmp_path)
    partial = run_segmented(SMALL, execution={'segment_steps': 10, 'stop_after_segment': 1},
                            root=tmp_path)
    with pytest.raises(ValueError, match='configuration differs'):
        run_segmented({**SMALL, 'mu': .4},
                      execution={'segment_steps': 10, 'resume_from': partial}, root=tmp_path)


def test_scan_members_and_analysis_share_the_new_fields(tmp_path):
    batch_path = run_scan({'base': SMALL, 'members': [{'mu': .2}, {'mu': .5}]}, root=tmp_path)
    batch = json.loads(batch_path.read_text())
    assert batch['status'] == 'complete'
    summary = analyze_sources([__import__('experiments.inverse_cascade.analyze', fromlist=['run_source'])
                               .run_source(Path(v['run'])) for v in batch['members']],
                              tmp_path/'scanreport', burn_in=.1, block=.06, K=2)
    assert {run_['mu'] for run_ in summary['runs']} == {.2, .5}
    assert all(run_['stationarity']['E']['reason'] for run_ in summary['runs']
               if not run_['stationarity']['E']['passed'])


def test_plan_matrix_is_dry_run_and_costs_are_labelled():
    entries = matrix()
    assert entries and len({item['id'] for item in entries}) == len(entries)
    assert all(item['depends_on'] is None or isinstance(item['depends_on'], list) for item in entries)
    ids = {item['id'] for item in entries}
    assert {'qual_512_0005', 'steady512_mu020', 'crit512_mu022'} <= ids
    large = [item for item in entries if item['grid'].startswith('4096')]
    assert large and all(item['cost_basis'] == 'extrapolated' for item in large)
    measured = [item for item in entries if item['grid'].startswith('512')]
    assert measured and all(item['cost_basis'] == 'historical_local_measurement' for item in measured)
    cost = budget(512, 500., .005, 20)
    assert cost['steps'] == 100000 and cost['wall_hours'] > 0
    # Base configuration stays inside the paper's smallest published case.
    assert BASE['kf'] == 8 and BASE['Re_plus'] == 1400.


def test_interval_quality_requirements_are_explicit():
    # White noise with unit block that is shorter than 10 tau_int: must not qualify.
    interval = .1
    times = np.arange(0., 60., interval)
    noise = np.random.default_rng(11).standard_normal(len(times))
    short = block_statistics(times, noise, start=0., duration=2*interval)
    assert short['interval_quality']['requirements']['block_at_least_10_tau_int'] is False
    assert short['interval_quality']['qualified'] is False
    long_block = block_statistics(times, noise, start=0., duration=5.)
    assert long_block['interval_quality']['block_over_tau_int'] > 10.
    assert long_block['interval_quality']['requirements']['at_least_20_blocks'] is False
    assert long_block['interval_quality']['qualified'] is False
    assert long_block['interval_quality']['note'].startswith('operational')


def test_null_ratios_carry_a_reason(tmp_path):
    control = run({**SMALL, 'mu': 0.}, root=tmp_path)
    summary = analyze([control], tmp_path/'nulls', burn_in=.1, block=.06, K=2)
    q_budget = summary['runs'][0]['budgets']['Q']
    assert q_budget['large_over_injection'] is None
    assert 'zero' in q_budget['notes']['large_over_injection']
    # The HD control has zero Q dissipation too, so that fraction is also null with a reason.
    assert q_budget['large_fraction_of_dissipation'] is None
    assert 'zero' in q_budget['notes']['large_fraction_of_dissipation']

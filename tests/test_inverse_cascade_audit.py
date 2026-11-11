"""Regression feedback loop for the pre-production inverse-cascade audit."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.inverse_cascade.config import effective_config
from experiments.inverse_cascade.plan import BASE, entry, matrix, validate_approval, validate_matrix
from experiments.records import digest, provenance
from experiments.inverse_cascade.stats import integrated_correlation_time, screen, stationarity_screen
from experiments.inverse_cascade.segments import run_segmented


SMALL = {'nx': 24, 'ny': 24, 'kf': 2, 'T': .12, 'dt': .01,
         'sample_every': 2, 'mu': .3, 'gamma': 1000.}


def test_scale_matrix_sets_actual_paper_parameters():
    expected = {16: (1024, 7.4e5), 32: (2048, 2.3e7), 64: (4096, 7.6e8)}
    for item in matrix():
        if item['stage'] != 'scale':
            continue
        kf = int(item['id'].split('_')[0][2:])
        c = effective_config({**BASE, **item['config']})
        n, re_minus = expected[kf]
        assert (c['kf'], c['nx'], c['ny'], c['Re_minus']) == (kf, n, n, re_minus)
        assert c['Re_plus'] == 1400.


def test_plan_approval_binds_config_code_and_dependency_evidence(tmp_path: Path):
    qualification = entry('qual', 'test', 'qualification', SMALL, '')
    production = entry('steady', 'test', 'steady', SMALL, '', ('qual',))
    entries = [qualification, production]
    validate_matrix(entries)
    evidence = tmp_path/'review.md'
    evidence.write_text('Explicit resource benchmark and numerical review fixture, not production evidence.')
    reference = {'path': evidence.name, 'sha256': digest(evidence)}
    approval = {'approval_schema_version': 1, 'reviewed_by': 'test reviewer',
                'entry_id': production['id'], 'config_sha256': production['config_sha256'],
                'code_sha256': provenance()['code_sha256'],
                'resource_review': {'passed': True, 'evidence': [reference]},
                'dependencies': [{'entry_id': 'qual', 'config_sha256': qualification['config_sha256'],
                                  'checks': {'numerical': True}, 'evidence': [reference]}]}
    path = tmp_path/'approval.json'
    path.write_text(json.dumps(approval))
    validate_approval(production, path, entries)
    for key, replacement in (('dependencies', []), ('code_sha256', 'stale'),
                             ('config_sha256', 'wrong'), ('resource_review', {'passed': False})):
        path.write_text(json.dumps({**approval, key: replacement}))
        with pytest.raises(ValueError):
            validate_approval(production, path, entries)
    path.write_text(json.dumps(approval))
    evidence.write_text('changed after review')
    with pytest.raises(ValueError, match='checksum'):
        validate_approval(production, path, entries)


def test_execute_without_approval_never_imports_runner(tmp_path: Path, monkeypatch):
    import sys
    from experiments.inverse_cascade import plan
    monkeypatch.setattr(sys, 'argv', ['plan', '--execute', '--only', 'qual', '--output', str(tmp_path/'out')])
    monkeypatch.setattr(plan, 'matrix', lambda: [entry('qual', 'test', 'qualification', SMALL, '')])
    monkeypatch.setattr('experiments.inverse_cascade.segments.run_segmented',
                        lambda *a, **k: pytest.fail('must not compute before approval'))
    with pytest.raises(SystemExit):
        plan.main()
    assert not (tmp_path/'out').exists()


def test_geyer_ar1_matches_analytic_time_not_half_pair_sum():
    phi, interval = .8, .1
    rng = np.random.default_rng(927)
    from scipy.signal import lfilter
    values = lfilter([1.], [1., -phi], rng.normal(size=100_000))[1000:]
    estimated = integrated_correlation_time(values, interval)
    # For rho(lag)=phi**lag: tau/dt=1/2+phi/(1-phi)=4.5.
    expected = interval*(1+phi)/(2*(1-phi))
    assert estimated['status'] == 'ok'
    assert estimated['tau_int'] == pytest.approx(expected, rel=.15)


def test_near_zero_flux_drift_uses_injection_reference():
    noise = np.random.default_rng(35).normal(size=2000)
    noise -= noise.mean()
    flux = np.r_[noise, noise+.03]
    times = np.arange(4000)*.1
    result = screen(times, flux, start=-.1, injection_reference=1., storage_rate=0.)
    assert result['conditions']['drift']['fraction_over_injection'] == pytest.approx(.03)
    assert result['passed']


def test_two_window_screen_checks_powers_and_dissipations():
    times = np.arange(4001)*.1
    noise = np.random.default_rng(81).normal(scale=.1, size=len(times))
    kwargs = {'injection': np.ones(len(times)), 'small': np.full(len(times), .5),
              'large': np.full(len(times), .5), 'energy': np.ones(len(times)),
              'start': 0., 'block': 2.}
    good = stationarity_screen(times, noise, **kwargs)
    json.dumps(good, allow_nan=False)
    assert good['passed'], good
    assert len(good['windows']) == 2
    assert good['windows'][0]['end'] == good['windows'][1]['start']
    assert good['statistics_qualified']
    for key in ('injection', 'small', 'large'):
        drifting = np.ones(len(times))
        drifting[1001:2001] *= 1.3
        bad = stationarity_screen(times, noise, **{**kwargs, key: drifting})
        assert not bad['passed'], key
    # Short data cannot silently stand in for the required two evaluation windows.
    short = stationarity_screen(times[:20], noise[:20], **{
        **kwargs, **{k: kwargs[k][:20] for k in ('injection', 'small', 'large', 'energy')}})
    assert not short['passed'] and 'insufficient' in short['reason']


def test_interval_centre_excludes_an_incomplete_tail():
    from experiments.inverse_cascade.analyze import block_statistics, _ratio
    times = np.arange(23.)
    values = np.r_[np.zeros(21), 100., 100.]
    result = block_statistics(times, values, start=0., duration=5.)
    assert result['mean'] > 0
    assert result['ci95_mean'] == 0
    assert result['ci95_halfwidth'] == 0
    assert result['tail_dropped_t0'] == 2.
    assert _ratio(1., -1.) is None


@pytest.mark.parametrize('segment_steps', [1, 3, 5])
def test_sampling_phase_and_empty_segments_match_continuous(tmp_path: Path, segment_steps: int):
    import h5py
    from experiments.inverse_cascade.run import run
    from experiments.inverse_cascade.chain import load_chain
    cfg = {**SMALL, 'sample_every': 4}
    reference = run(cfg, root=tmp_path)
    chain = run_segmented(cfg, execution={'segment_steps': segment_steps}, root=tmp_path)
    got = load_chain(chain, require_complete=True)
    with h5py.File(reference/'results.h5') as d:
        np.testing.assert_array_equal(got['sample_times'], d['spectra/times'][:])
        np.testing.assert_allclose(got['arrays']['spectra']['Pi_E'], d['spectra/Pi_E'][:], atol=1e-14)
        np.testing.assert_array_equal(got['arrays']['stages']['r'], d['stages/r'][:])


@pytest.mark.parametrize('field,value', [('sample_count', 999), ('config_sha256', 'bad'),
                                         ('steps_after', 999), ('t', 999.)])
def test_checkpoint_metadata_is_checked_before_resume(tmp_path: Path, field: str, value):
    directory = run_segmented(SMALL, execution={'segment_steps': 4, 'stop_after_segment': 1},
                              root=tmp_path)
    path = directory/'checkpoints'/'0000.json'
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        run_segmented(SMALL, execution={'resume_from': str(directory)}, root=tmp_path)


def test_wall_budget_pauses_only_after_a_committed_segment(tmp_path: Path, monkeypatch):
    from experiments.inverse_cascade import segments
    calls = iter([0., 0., 2.])
    monkeypatch.setattr(segments, 'monotonic', lambda: next(calls), raising=False)
    directory = run_segmented(SMALL, execution={'segment_steps': 4, 'max_wall_seconds': 1.}, root=tmp_path)
    chain = json.loads((directory/'chain.json').read_text())
    assert chain['status'] == 'running' and chain['accepted_steps'] == 4
    assert chain['stop_reason'] == 'max_wall_seconds'


def test_time_check_tolerates_accumulation_but_detects_a_missing_step():
    from experiments.inverse_cascade.chain import _equal_time
    h = .005/(8**0.5)  # dt*=0.005 at kf=8: the long-window step
    t = 0.
    for _ in range(200_000):
        t += h
    # Accumulated float error must not be mistaken for a step-count error...
    assert abs(t-200_000*h) > 1e-12*max(1., t)
    assert _equal_time(t, 200_000*h)
    # ...while being off by one step stays detectable.
    assert not _equal_time(t, 199_999*h)
    assert not _equal_time(t, 200_001*h)


def test_resume_binds_numerical_source_and_tolerates_a_reader_fix(tmp_path: Path, monkeypatch):
    from experiments.inverse_cascade import segments
    directory = run_segmented(SMALL, execution={'segment_steps': 4, 'stop_after_segment': 1}, root=tmp_path)
    stored = json.loads((directory/'chain.json').read_text())
    assert stored['numerical_sha256'] == segments.numerical_digest()
    checkpoint = json.loads((directory/'checkpoints'/'0000.json').read_text())
    assert checkpoint['numerical_sha256'] == stored['numerical_sha256']
    assert checkpoint['harness_sha256']
    source = provenance()
    # A reader-only fix changes the aggregate provenance hash but not the numbers.
    monkeypatch.setattr(segments, 'provenance', lambda: {**source, 'code_sha256': 'reader-fix'})
    run_segmented(SMALL, execution={'resume_from': directory, 'stop_after_segment': 1}, root=tmp_path)
    assert json.loads((directory/'chain.json').read_text())['accepted_steps'] == 8
    # Changing the numerical source must still refuse continuation.
    monkeypatch.setattr(segments, 'numerical_digest', lambda: 'changed-numerics')
    with pytest.raises(ValueError, match='numerical source changed'):
        run_segmented(SMALL, execution={'resume_from': directory}, root=tmp_path)


def test_legacy_chain_needs_explicit_migration_which_is_recorded(tmp_path: Path, monkeypatch):
    from experiments.inverse_cascade import segments
    from experiments.inverse_cascade.segments import migrate_identity
    directory = run_segmented(SMALL, execution={'segment_steps': 4, 'stop_after_segment': 1}, root=tmp_path)
    chain = json.loads((directory/'chain.json').read_text())
    chain.pop('numerical_sha256')
    (directory/'chain.json').write_text(json.dumps(chain))
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['chain_sha256'] = digest(directory/'chain.json')
    (directory/'manifest.json').write_text(json.dumps(manifest))
    # A pre-split chain comes from a different aggregate code version.
    source = provenance()
    monkeypatch.setattr(segments, 'provenance', lambda: {**source, 'code_sha256': 'older-harness'})
    with pytest.raises(ValueError, match='no numerical digest'):
        run_segmented(SMALL, execution={'resume_from': directory}, root=tmp_path)
    with pytest.raises(ValueError, match='reason'):
        migrate_identity(directory, reason='  ', reviewer='test')
    result = migrate_identity(directory, reason='read-only validator tolerance fix',
                              reviewer='audit test')
    assert result['validated_steps'] == 4
    migrated = json.loads((directory/'chain.json').read_text())
    assert migrated['numerical_sha256'] == result['numerical_sha256']
    assert migrated['identity'] == chain['identity']  # original identity is immutable
    record, = migrated['identity_migration']
    assert record['from_aggregate_sha256'] == chain['identity']
    assert record['reason'] == 'read-only validator tolerance fix'
    assert record['harness_sha256_at_migration'] == 'older-harness'
    assert 'segment payloads' in record['does_not_change']
    with pytest.raises(ValueError, match='already carries'):
        migrate_identity(directory, reason='again', reviewer='audit test')
    run_segmented(SMALL, execution={'resume_from': directory}, root=tmp_path)
    assert json.loads((directory/'chain.json').read_text())['status'] == 'complete'


def test_migration_refuses_a_chain_whose_segments_do_not_validate(tmp_path: Path):
    import h5py
    from experiments.inverse_cascade.segments import migrate_identity
    directory = run_segmented(SMALL, execution={'segment_steps': 4, 'stop_after_segment': 1}, root=tmp_path)
    with h5py.File(directory/'segments'/'0000.h5', 'a') as data:
        values = data['diagnostics/total_energy'][:]
        values[1] = np.nan
        data['diagnostics/total_energy'][...] = values
    with pytest.raises(ValueError, match='checksum'):
        migrate_identity(directory, reason='try', reviewer='audit test')
    assert 'identity_migration' not in json.loads((directory/'chain.json').read_text())
    # Even with the checksums re-bound, a frozen chain cannot be migrated.
    _rebind_for_semantic_test(directory)
    with pytest.raises(ValueError, match='finite'):
        migrate_identity(directory, reason='try', reviewer='audit test')


def _rebind_for_semantic_test(directory: Path, *, repair_parents: bool = False) -> None:
    """Rehash fixtures to exercise semantic validation past the transport checksum."""
    from experiments.inverse_cascade.chain import chain_digest
    chain = json.loads((directory/'chain.json').read_text())
    parent = None
    for segment in chain['segments']:
        segment['sha256'] = digest(directory/segment['file'])
        checkpoint_path = directory/'checkpoints'/f"{segment['index']:04d}.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint['segment_sha256'] = segment['sha256']
        if repair_parents:
            checkpoint['parent_checkpoint_sha256'] = parent
        checkpoint_path.write_text(json.dumps(checkpoint))
        segment['checkpoint_sha256'] = digest(checkpoint_path)
        parent = segment['checkpoint_sha256']
    (directory/'chain.json').write_text(json.dumps(chain))
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['chain_sha256'] = digest(directory/'chain.json')
    if chain['status'] == 'complete':
        manifest['results_sha256'] = chain_digest(chain)
    (directory/'manifest.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('field,value', [('parent_checkpoint_sha256', 'wrong-parent'),
                                         ('sample_phase', 999), ('sample_count', 999),
                                         ('config_sha256', 'wrong'), ('t', 999.)])
def test_chain_semantics_reject_rehashed_wrong_checkpoint(tmp_path: Path, field: str, value):
    from experiments.inverse_cascade.chain import load_chain
    directory = run_segmented(SMALL, execution={'segment_steps': 4}, root=tmp_path)
    path = directory/'checkpoints'/'0001.json'
    checkpoint = json.loads(path.read_text())
    checkpoint[field] = value
    path.write_text(json.dumps(checkpoint))
    _rebind_for_semantic_test(directory)
    with pytest.raises(ValueError, match='Checkpoint'):
        load_chain(directory, require_complete=True)


@pytest.mark.parametrize('dataset', ['stages/r', 'final/magnetic', 'spectra/Pi_E'])
def test_chain_rejects_nonfinite_payload_even_after_rehash(tmp_path: Path, dataset: str):
    import h5py
    from experiments.inverse_cascade.chain import load_chain
    directory = run_segmented(SMALL, execution={'segment_steps': 4}, root=tmp_path)
    with h5py.File(directory/'segments'/'0000.h5', 'a') as data:
        values = data[dataset][:]
        values.flat[0] = np.nan
        data[dataset][...] = values
    _rebind_for_semantic_test(directory)
    with pytest.raises(ValueError, match='finite'):
        load_chain(directory, require_complete=True)


def test_failed_gate_retains_evidence_but_commits_no_bad_segment(tmp_path: Path):
    bad = {**SMALL, 'gamma': 1., 'coupling_tolerance': 1e-25}
    with pytest.raises(RuntimeError, match='coupling'):
        run_segmented(bad, execution={'segment_steps': 4}, root=tmp_path)
    directory, = (tmp_path/'runs'/'inverse_cascade_chains').iterdir()
    chain = json.loads((directory/'chain.json').read_text())
    assert chain['status'] == 'failed' and chain['segments'] == []
    assert (directory/'failed'/'0000.h5').exists()
    with pytest.raises(ValueError, match='failed'):
        run_segmented(bad, execution={'resume_from': directory}, root=tmp_path)


def test_full_field_boundary_and_source_identity_are_verified(tmp_path: Path, monkeypatch):
    import h5py
    from experiments.inverse_cascade.chain import load_chain
    from experiments.inverse_cascade import segments
    directory = run_segmented(SMALL, execution={'segment_steps': 4}, root=tmp_path)
    with h5py.File(directory/'segments'/'0001.h5', 'a') as data:
        data['initial/u'][0, 0] += .01
    _rebind_for_semantic_test(directory, repair_parents=True)
    with pytest.raises(ValueError, match='accepted-state continuity'):
        load_chain(directory, require_complete=True)
    partial = run_segmented(SMALL, execution={'segment_steps': 4, 'stop_after_segment': 1}, root=tmp_path)
    source = provenance()
    # A reader-only harness change is allowed; a numerical change is not.
    monkeypatch.setattr(segments, 'provenance', lambda: {**source, 'code_sha256': 'reader-only'})
    run_segmented(SMALL, execution={'resume_from': partial, 'stop_after_segment': 1}, root=tmp_path)
    monkeypatch.setattr(segments, 'numerical_digest', lambda: 'changed-numerics')
    with pytest.raises(ValueError, match='numerical source changed'):
        run_segmented(SMALL, execution={'resume_from': partial}, root=tmp_path)


def test_lowpass_window_uses_physical_time_trapezoidal_integrals():
    from experiments.inverse_cascade.analyze import _lowpass_budget
    times = np.arange(101)*.2  # physical time; t0=2, not unity
    factors = np.array([.2, .6, 1.])
    # Exact trapezoidal primitive of 3*t**2 at spacing h:
    primitive = times**3+.2**2*times/2
    data = {'spectra/times': times}
    for q, scale in [('E', 1.), ('Q', 3.)]:
        injection = scale*3*times[:, None]**2*factors
        low = scale*primitive[:, None]*factors
        data.update({'spectra/'+q+'_low': low, 'spectra/Pi_'+q: np.zeros_like(low),
                     'spectra/power_'+q+'_low': injection,
                     'spectra/diss_'+q+'_plus_low': np.zeros_like(low),
                     'spectra/diss_'+q+'_minus_low': np.zeros_like(low)})
    for cutoff in range(3):
        result = _lowpass_budget(data, {'t0': 2.}, start=1., cutoff=cutoff)
        assert result['status'] == 'ok'
        for q in ('E', 'Q'):
            assert result[q]['max_abs_window_defect_over_total_injection'] < 1e-14
            assert all(abs(value) < 1e-14 for value in result[q]['window_defect_over_total_injection_all_K'])
    # A defect outside the selected K must still fail the all-cutoff gate.
    data['spectra/Pi_Q'][:, 2] += 100.
    assert not _lowpass_budget(data, {'t0': 2.}, start=1., cutoff=0)['Q']['passed']


def test_legacy_chain_is_readable_but_not_resumable(tmp_path: Path):
    from experiments.inverse_cascade.chain import load_chain, chain_digest
    directory = run_segmented(SMALL, execution={'segment_steps': 4, 'stop_after_segment': 1}, root=tmp_path)
    chain = json.loads((directory/'chain.json').read_text())
    chain.pop('chain_schema_version')
    (directory/'chain.json').write_text(json.dumps(chain))
    loaded = load_chain(directory)
    assert loaded['chain']['accepted_steps'] == 4
    with pytest.raises(ValueError, match='Legacy chain'):
        run_segmented(SMALL, execution={'resume_from': directory}, root=tmp_path)
    # Legacy completed chains still verify their historical ordered-segment digest.
    chain['total_steps'] = 4
    # Do not falsify completion: the original T implies 12 steps, and the reader must reject it.
    chain['status'] = 'complete'
    (directory/'chain.json').write_text(json.dumps(chain))
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest.update(status='complete', results_sha256=chain_digest(chain))
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='terminal step'):
        load_chain(directory, require_complete=True)


def test_failed_chain_cannot_resume_or_become_complete(tmp_path: Path):
    directory = run_segmented(SMALL, execution={'segment_steps': 4, 'stop_after_segment': 1},
                              root=tmp_path)
    # Model an already recorded gate failure, not an intentional segment-boundary pause.
    for name in ('chain.json', 'manifest.json'):
        path = directory/name
        metadata = json.loads(path.read_text())
        metadata.update(status='failed', error='coupling gate failed')
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='failed'):
        run_segmented(SMALL, execution={'resume_from': str(directory)}, root=tmp_path)

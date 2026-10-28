import json

import h5py
import numpy as np
import pytest

from experiments.channel_ns.analyze import analyze, read_fields
from experiments.channel_ns.problems import SCHEMES
from experiments.channel_ns.run import effective_config, load_channel, run
from experiments.workflow import digest, write_json
from solver.schemes.registry import scheme_info


@pytest.mark.parametrize('name', SCHEMES)
def test_fixed_prescribed_roundtrip_reuse_and_read_only_analysis(tmp_path, name, monkeypatch):
    cfg = {'case': 'discrete_mms', 'nx': 10, 'ny': 8, 'scheme': name,
           'T': .03, 'steps': [.01, .02], 'snapshots': [0., .015, .03]}
    directory = run(cfg, root=tmp_path)
    stored, saved = load_channel(directory, require_complete=True)
    assert stored['actual_steps'] == cfg['steps']
    assert run(cfg, root=tmp_path) == directory
    assert run(cfg, root=tmp_path, rerun=True) != directory
    with h5py.File(directory/'results.h5') as data:
        info = scheme_info(name)
        stages = info.n_stages
        assert data['final/u'].shape == (8, 11) and data['final/v'].shape == (9, 10)
        assert data['final/stage_pressure'].shape == (stages, 8, 10)
        assert data['roots/candidates'].shape == (2, stages, info.n_roots)
        np.testing.assert_array_equal(data['final/u'][:, 0], data['final/u'][:, -1])
        np.testing.assert_array_equal(data['final/v'][[0, 8], :], 0.)
    import experiments.channel_ns.run as runner
    monkeypatch.setattr(runner, 'build_model', lambda cfg: (_ for _ in ()).throw(AssertionError('analysis constructed model')))
    report = analyze(directory, root=tmp_path)
    summary = json.loads((report/'analysis.json').read_text())
    assert summary['status'] == 'complete'
    assert summary['velocity_L2_error'] < (1e-3 if scheme_info(name).order == 1 else 1e-5)
    _, fields = read_fields(directory)
    assert fields['speed'].shape == (8, 10) and fields['vorticity'].shape == (7, 10)
    assert fields['flow_rates'].shape == (10,)
    assert summary['flow_rate_spread'] < 1e-12
    with h5py.File(directory/'results.h5', 'a') as data: data['final/u'][0, 0] += 1.
    with pytest.raises(ValueError, match='checksum'): analyze(directory, root=tmp_path)


def test_adaptive_attempts_joint_control_and_failed_prefix(tmp_path):
    cfg = {'case': 'discrete_mms', 'nx': 10, 'ny': 8, 'T': .1, 'mode': 'adaptive',
           'initial_step': .1, 'scalar_atol': 1e-4,
           'controller': {'kind': 'I', 'atol': 1e-10, 'rtol': 1e-7}}
    directory = run(cfg, root=tmp_path)
    _, saved = load_channel(directory, require_complete=True)
    assert saved['metrics']['rejected_steps'] > 0
    with h5py.File(directory/'results.h5') as data:
        assert data['times'][-1] == .1
        assert np.count_nonzero(data['attempts/accepted'][:]) == len(data['dt'])
        np.testing.assert_array_equal(data['attempts/error'][:],
                                      np.maximum(data['attempts/error_u'][:], data['attempts/error_r'][:]))
    cfg['controller']['max_steps'] = 1
    failed = run(cfg, root=tmp_path)
    _, saved = load_channel(failed)
    assert saved['status'] == 'failed'
    summary = json.loads((analyze(failed, root=tmp_path)/'analysis.json').read_text())
    assert summary['status'] == 'failed' and summary['accepted_steps'] == 1
    assert summary['final_time'] < .1
    assert run(cfg, root=tmp_path) != failed


def test_poiseuille_startup_and_perturbation_parameters_reach_model(tmp_path):
    from experiments.channel_ns.problems import build_model
    fields = []
    for driving in [.1, .2]:
        cfg = effective_config({'case': 'poiseuille', 'driving': driving, 'nx': 7, 'ny': 6})
        model, initial, _ = build_model(cfg)
        fields.append(model.vector(initial))
    np.testing.assert_allclose(fields[1], 2*fields[0])
    for case in ['poiseuille', 'startup', 'perturbed', 'mms']:
        directory = run({'case': case, 'nx': 10, 'ny': 8, 'T': .02, 'dt': .01}, root=tmp_path)
        summary = json.loads((analyze(directory, root=tmp_path)/'analysis.json').read_text())
        assert summary['status'] == 'complete'
        if case == 'poiseuille': assert summary['discrete_equilibrium_L2_error'] < 1e-12


def test_channel_reader_rejects_wrong_layout_even_with_updated_checksum(tmp_path):
    directory = run({'case': 'startup', 'nx': 7, 'ny': 6, 'T': .01}, root=tmp_path)
    with h5py.File(directory/'results.h5', 'a') as data: data.attrs['boundary'] = 'periodic_xy'
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['results_sha256'] = digest(directory/'results.h5')
    write_json(directory/'manifest.json', manifest)
    with pytest.raises(ValueError, match='representation'): load_channel(directory)


@pytest.mark.parametrize('cfg', [{'boundary': 'periodic_xy'}, {'backend': 'unknown'},
                                {'mode': 'adaptive', 'dt': .01},
                                {'scheme': 'sdirk2', 'mode': 'adaptive', 'scalar_atol': 1e-4},
                                {'cache_size': 0}, {'r0': .1, 'case': 'discrete_mms'}])
def test_invalid_channel_configuration(cfg):
    with pytest.raises(ValueError): effective_config(cfg)


def test_validation_persists_fields_and_analysis_rejects_corrupt_member(tmp_path):
    from experiments.channel_ns.validate import analyze_validation, validate
    directory = validate({'schemes': ['sdirk3'], 'gammas': [1.], 'temporal_counts': [40, 80],
                          'spatial_sizes': [8, 16], 'smoke_cases': []}, root=tmp_path)
    report = analyze_validation(directory, root=tmp_path)
    summary = json.loads((report/'analysis.json').read_text())
    assert summary['expected_velocity_orders_observed']
    children = list(directory.rglob('results.h5'))
    analyze_validation(directory, root=tmp_path)
    assert list(directory.rglob('results.h5')) == children
    with h5py.File(children[0], 'a') as data: data['final/u'][0, 0] += 1.
    with pytest.raises(ValueError, match='checksum'): analyze_validation(directory, root=tmp_path)


def test_fft_backend_run_load_and_analyze(tmp_path):
    directory = run({'case': 'discrete_mms', 'backend': 'fft', 'nx': 10, 'ny': 8,
                     'T': .02, 'dt': .01}, root=tmp_path)
    cfg, manifest = load_channel(directory, require_complete=True)
    assert cfg['backend'] == 'fft' and manifest['metrics']['factorizations'] > 0
    summary = json.loads((analyze(directory, root=tmp_path)/'analysis.json').read_text())
    assert summary['velocity_L2_error'] < 1e-6


def test_benchmark_saved_pairs_and_read_only_analysis(tmp_path):
    from experiments.channel_ns.benchmark import analyze_benchmark, benchmark
    directory = benchmark({'grids': [[8, 6]], 'schemes': ['sdirk3'], 'repeats': 1, 'steps': 2}, root=tmp_path)
    report = analyze_benchmark(directory, root=tmp_path)
    summary = json.loads((report/'analysis.json').read_text())
    assert summary['max_velocity_L2_backend_difference'] < 1e-12
    assert len(summary['timings']) == 1
    files = list(directory.rglob('results.h5'))
    analyze_benchmark(directory, root=tmp_path)
    assert list(directory.rglob('results.h5')) == files

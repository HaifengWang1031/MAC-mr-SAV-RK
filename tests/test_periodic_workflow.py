import json

import h5py
import numpy as np
import pytest

from experiments.periodic_ns.analyze import analyze, read_fields
from experiments.periodic_ns.run import effective_config, load_periodic, run
from experiments.periodic_ns.problems import SCHEMES
from solver.schemes.registry import scheme_info


@pytest.mark.parametrize('name', SCHEMES)
def test_fixed_prescribed_save_load_analyze_reuse(tmp_path, name):
    cfg = {'nx': 18, 'ny': 15, 'scheme': name, 'T': .03, 'steps': [.01, .02], 'snapshots': [0., .015, .03]}
    directory = run(cfg, root=tmp_path)
    stored, manifest = load_periodic(directory, require_complete=True)
    assert stored['actual_steps'] == cfg['steps']
    assert run(cfg, root=tmp_path) == directory
    assert run(cfg, root=tmp_path, rerun=True) != directory
    with h5py.File(directory/'results.h5') as data:
        stages = scheme_info(name).n_stages
        assert data['final/u'].shape == data['final/v'].shape == (15, 18)
        assert data['final/stage_pressure'].shape == (stages, 15, 18)
        assert data['roots/candidates'].shape[:2] == (2, stages)
        assert data['snapshots/requested_times'][:].tolist() == cfg['snapshots']
    report = analyze(directory, root=tmp_path)
    summary = json.loads((report/'analysis.json').read_text())
    assert summary['status'] == 'complete'
    assert summary['velocity_L2_error'] < (.01 if scheme_info(name).order == 1 else .001)
    _, fields = read_fields(directory)
    assert fields['vorticity'].shape == (15, 18)
    # Analysis never creates runs; corruption is rejected by both read and reuse.
    count = len(list((tmp_path/'runs/periodic_ns').iterdir()))
    analyze(directory, root=tmp_path)
    assert len(list((tmp_path/'runs/periodic_ns').iterdir())) == count
    with h5py.File(directory/'results.h5', 'a') as data:
        data['final/u'][0, 0] += 1.
    with pytest.raises(ValueError, match='checksum'):
        analyze(directory, root=tmp_path)


def test_adaptive_attempts_and_failed_prefix_persist(tmp_path):
    cfg = {'nx': 18, 'ny': 15, 'T': .1, 'mode': 'adaptive', 'initial_step': .1,
           'controller': {'kind': 'I', 'atol': 1e-10, 'rtol': 1e-7}}
    directory = run(cfg, root=tmp_path)
    _, manifest = load_periodic(directory, require_complete=True)
    assert manifest['metrics']['rejected_steps'] > 0
    with h5py.File(directory/'results.h5') as data:
        assert data['times'][-1] == .1
        assert np.count_nonzero(data['attempts/accepted'][:]) == len(data['dt'])
        assert data['attempts/error_u'].size == data['attempts/reason'].size
    cfg['controller']['max_steps'] = 1
    failed = run(cfg, root=tmp_path)
    _, manifest = load_periodic(failed)
    assert manifest['status'] == 'failed'
    with h5py.File(failed/'results.h5') as data:
        assert len(data['dt']) == 1 and data['times'][-1] < .1
    summary = json.loads((analyze(failed, root=tmp_path)/'analysis.json').read_text())
    assert summary['status'] == 'failed' and summary['accepted_steps'] == 1
    assert run(cfg, root=tmp_path) != failed


@pytest.mark.parametrize('cfg', [{'boundary': 'no_slip'}, {'field_layout': 'mac'},
                                {'mode': 'adaptive', 'dt': .1},
                                {'mode': 'adaptive', 'scheme': 'sdirk2', 'scalar_atol': 1e-5},
                                {'case': 'kolmogorov', 'ny': 12, 'k_f': 4}])
def test_periodic_config_rejects_incompatible_interfaces(cfg):
    with pytest.raises(ValueError):
        effective_config(cfg)


def test_validation_benchmark_saved_members_and_read_only_summary(tmp_path):
    from experiments.periodic_ns.validate import validate
    from experiments.periodic_ns.analyze import analyze_validation
    directory = validate({'temporal_size': 18, 'temporal_counts': [10, 20],
                          'schemes': ['sdirk3'], 'gammas': [1.], 'spatial_sizes': [12, 18],
                          'benchmark_sizes': [16], 'repeats': 1, 'operator_repeats': 2}, root=tmp_path)
    summary = json.loads((analyze_validation(directory, root=tmp_path)/'analysis.json').read_text())
    assert summary['expected_velocity_orders_observed']
    assert len(summary['timings']) == 2 and len(summary['profiles']) == 4
    before = list(directory.rglob('results.h5'))
    analyze_validation(directory, root=tmp_path)
    assert list(directory.rglob('results.h5')) == before
    with h5py.File(before[0], 'a') as data:
        data['final/u'][0, 0] += 1.
    with pytest.raises(ValueError, match='checksum'):
        analyze_validation(directory, root=tmp_path)

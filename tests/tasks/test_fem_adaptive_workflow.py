"""Adaptive closed FEM records share attempt semantics without changing old fixed configs."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.fem_ns.run import effective_config, load_fem, run
from experiments.fem_ns.reconstruct import reconstruct
from experiments.fem_ns.analyze import analyze
from experiments.workflow import digest


@pytest.mark.parametrize('backend', ['direct', 'recycled'])
@pytest.mark.parametrize(('scheme', 'kind'), [('sdirk2', 'I'), ('sdirk3_mrsav', 'PI')])
def test_adaptive_fem_run_load_and_readonly_analysis(backend: str, scheme: str, kind: str, tmp_path: Path):
    directory = run({'nx': 6, 'ny': 4, 'T': .05, 'mode': 'adaptive', 'initial_step': .03,
                     'stokes_backend': backend, 'scheme': scheme,
                     'controller': {'kind': kind, 'rtol': 1e-5, 'max_step': .03},
                     'scalar_atol': 1e-6 if scheme.endswith('_mrsav') else None,
                     'snapshots': [0., .017, .05]}, root=tmp_path)
    cfg, manifest = load_fem(directory, require_complete=True)
    with h5py.File(directory/'results.h5', 'r') as data:
        times, accepted = data['times'][:], data['attempts/accepted'][:]
        assert times[-1] == cfg['T'] and sum(accepted) == len(times)-1
        assert len(accepted) >= sum(accepted) and np.all(data['attempts/error'][:][accepted] <= 1.)
        nearest = [times[int(np.argmin(abs(times-t)))] for t in cfg['snapshots']]
        np.testing.assert_array_equal(data['snapshots/actual_times'][:], nearest)
    assert manifest['metrics']['stokes_max_coefficient_residual'] < 1e-9
    fields = reconstruct(directory, np.array([[.3, 1.4], [.2, .7]]))
    assert fields['time'] == .05 and np.isfinite(fields['velocity']).all()
    before = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    report = analyze(directory, root=tmp_path)
    assert (report/'attempts.csv').is_file()
    assert before == {p.name: digest(p) for p in directory.iterdir() if p.is_file()}


def test_old_fixed_effective_configuration_keys_remain_unchanged():
    cfg = effective_config({'nx': 4, 'ny': 3, 'T': .01, 'dt': .005})
    assert not {'mode', 'controller', 'initial_step', 'scalar_atol', 'stokes_backend', 'stokes_ordering',
                'stokes_factor_backend', 'stokes_threads'} & set(cfg)
    np.testing.assert_array_equal(cfg['actual_steps'], [.005, .005])


def test_adaptive_fem_failure_preserves_accepted_prefix(tmp_path: Path):
    directory = run({'nx': 4, 'ny': 3, 'T': .004, 'mode': 'adaptive', 'initial_step': .001,
                     'controller': {'kind': 'I', 'max_steps': 1}}, root=tmp_path)
    cfg, manifest = load_fem(directory)
    assert cfg['mode'] == 'adaptive' and manifest['status'] == 'failed'
    assert manifest['metrics']['accepted_steps'] == 1
    with pytest.raises(ValueError):
        load_fem(directory, require_complete=True)
    assert reconstruct(directory, np.array([[.3], [.2]]))['time'] == .001

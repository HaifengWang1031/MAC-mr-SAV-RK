"""Short public run/save workflow; not a production-qualification test."""
import json

import numpy as np

from experiments.torque_transition.preflight import run_probe


def test_probe_persists_nearest_samples_and_preserves_long_target(tmp_path):
    directory = tmp_path/'probe'
    row = run_probe(directory, nodes=20, case=4, duration=1.)
    assert row['status'] == 'complete' and row['final_t'] == 1.
    assert row['production_target']['T'] == 100000.
    assert row['production_target']['field_interval'] == 10.
    assert row['max_coupling_deviation'] < 1e-3
    samples = json.loads((directory/'scalar_samples.json').read_text())['samples']
    assert [s['request_t'] for s in samples] == [0., 1.]
    with np.load(directory/'history.npz') as history:
        for sample in samples:
            assert abs(sample['actual_t']-sample['request_t']) == min(
                abs(history['times']-sample['request_t']))
    with np.load(directory/'field_samples.npz') as fields:
        np.testing.assert_array_equal(fields['requests'], [0.])
    with np.load(directory/'final.npz') as final:
        assert np.isfinite(final['velocity']).all()

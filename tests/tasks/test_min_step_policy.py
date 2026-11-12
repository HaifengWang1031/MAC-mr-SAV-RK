"""Cylinder forced-floor errors and legacy configuration identity roundtrip."""
from pathlib import Path

import h5py
import numpy as np


def test_cylinder_floor_option_roundtrips_forced_errors_and_legacy_identity(tmp_path: Path):
    from experiments.cylinder_ns.run import run, load_cylinder
    from experiments.cylinder_ns.config import effective_config
    base = {'resolution': .25, 'T': .0025, 'initial_step': .002, 'scheme': 'sdirk3_mrsav',
            'scalar_atol': 1e-12,
            'controller': {'kind': 'PI', 'atol': 1e-13, 'rtol': 0., 'min_step': .001, 'max_step': .002}}
    assert 'min_step_policy' not in effective_config(base)
    directory = run({**base, 'min_step_policy': 'force'}, root=tmp_path)
    cfg, manifest = load_cylinder(directory)
    # The final short step retains its normal error check; prior forced steps
    # must remain readable even if that final short step is rejected.
    assert cfg['min_step_policy'] == 'force'
    with h5py.File(directory/'results.h5', 'r') as data:
        accepted, error, steps = (data['attempts/'+key][:] for key in ('accepted', 'error', 'step'))
        forced = accepted & (error > 1.)
        assert np.any(forced) and np.all(steps[forced] == .001)
        assert manifest['metrics']['forced_min_step_acceptances'] == int(np.count_nonzero(forced))
    legacy = run({'resolution': .25, 'T': .001, 'initial_step': .001}, root=tmp_path)
    assert 'min_step_policy' not in load_cylinder(legacy, require_complete=True)[0]

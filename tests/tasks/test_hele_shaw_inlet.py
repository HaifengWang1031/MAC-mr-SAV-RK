"""Airfoil inlet configuration, records, runtime limits and probe replay."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.hele_shaw_airfoil.config import effective_config


def test_inlet_config_identity_and_saved_public_workflow(tmp_path: Path):
    pytest.importorskip('gmsh')
    from experiments.hele_shaw_airfoil.run import run
    from experiments.hele_shaw_airfoil.records import load
    old = effective_config({})
    assert 'inlet_treatment' not in old
    with pytest.raises(ValueError, match='inlet_treatment'):
        effective_config({'inlet_treatment': 'unknown'})
    cfg = dict(inlet_treatment='energy_stable_pressure', gap_ratio=.3, layer_cells=1,
               surface_points=32, far_size=.6, T=.004, dt=.002, scheme='sdirk3_mrsav')
    new = effective_config(cfg)
    assert new['boundary'] != old['boundary']
    assert 'energy_stable_inlet' in new['pressure_semantics']
    directory = run(cfg, root=tmp_path)
    loaded, manifest, *_ = load(directory)
    assert loaded == new and manifest['status'] == 'complete'
    with h5py.File(directory/'results.h5') as f:
        assert f.attrs['boundary'] == new['boundary']
        assert f.attrs['inlet_treatment'] == 'energy_stable_pressure'
        assert np.isfinite(f['diagnostics/inlet_kinetic_energy_correction'][:]).all()
        assert f['times'][-1] == .004
    import json
    live = json.loads((directory/'progress.json').read_text())
    assert live['status'] == 'complete' and live['t'] == .004
    # Runtime limits stop at a fully accepted state; nothing is clipped.
    stopped = run({**cfg, 'velocity_limit': 1e-12}, root=tmp_path)
    _, failed, *_ = load(stopped, require_complete=False)
    assert failed['status'] == 'failed'
    assert 'Runtime qualification limit exceeded' in failed['error']
    with h5py.File(stopped/'results.h5') as f:
        assert len(f['times']) == 2 and f['times'][-1] == .002
    # A replay starts at a nonzero saved time; the target is an absolute time.
    from tools.probe_hele_shaw_inlet import probe
    adaptive = run({**cfg, 'mode': 'adaptive', 'dt': None, 'T': .006,
                    'initial_step': .002, 'scalar_atol': 1e-4,
                    'controller': {'max_step': .002}}, root=tmp_path)
    replay = probe(adaptive, inlet_treatment='energy_stable_pressure', replay_to=.008)
    assert replay['initial_time'] == .006 and replay['final_time'] == .008
    assert replay['passed']

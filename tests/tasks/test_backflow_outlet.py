"""Cylinder backflow boundary labels and reference qualification."""
from pathlib import Path

import h5py


def test_cylinder_backflow_roundtrip_has_distinct_boundary_and_reference_identity(tmp_path: Path):
    from experiments.cylinder_ns.run import run, load_cylinder
    from experiments.cylinder_ns.config import effective_config
    from experiments.cylinder_ns.benchmark import reference_comparison
    config = {'resolution': .25, 'T': .004, 'initial_step': .001, 'outlet_treatment': 'backflow_damped'}
    output = run(config, root=tmp_path)
    cfg, manifest = load_cylinder(output, require_complete=True)
    assert cfg['case'] == 'dfg2d2_backflow_damped'
    assert 'backflow_damped' in cfg['boundary'] and 'backflow_damped' in cfg['physical_pressure_semantics']
    assert not reference_comparison({}, cfg, complete=True)['canonical_physics']
    with h5py.File(output/'results.h5', 'r') as f:
        assert f.attrs['boundary'] == cfg['boundary']
        assert f['diagnostics/outlet_backflow_dissipation'].shape == f['times'].shape
    assert 'outlet_treatment' not in effective_config({'resolution': .25, 'T': .004})

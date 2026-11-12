"""PETSc identity and readonly loading for four research run families."""
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec('petsc4py') is None,
                                reason='optional matching PETSc/petsc4py')


@pytest.mark.parametrize('family', ['fem', 'cylinder', 'wedge', 'hele_shaw'])
def test_public_run_load_keeps_explicit_backend_identity_without_analysis_solver(tmp_path, monkeypatch, family: str):
    from experiments.fem_ns.run import run as fem_run, load_fem
    from experiments.cylinder_ns.run import run as cylinder_run, load_cylinder
    from experiments.wedge_cavity.run import run as wedge_run
    from experiments.wedge_cavity.report import open_run
    from experiments.hele_shaw_airfoil.run import run as hele_run
    from experiments.hele_shaw_airfoil.records import load as hele_load
    from solver.fem import multigrid
    mass_backend = 'metis' if family=='wedge' else 'cg'
    options = dict(stokes_backend='petsc', mass_backend=mass_backend, stokes_amg_type='gamg', fem_load_mode='weak',
                   T=.004, dt=.002, scheme='sdirk3_mrsav')
    if family == 'fem':
        directory = fem_run({**options, 'nx': 4, 'ny': 3}, root=tmp_path)
        reader = lambda: load_fem(directory, require_complete=True)
    elif family == 'cylinder':
        directory = cylinder_run({**options, 'resolution': .25, 'mode': 'fixed'}, root=tmp_path)
        reader = lambda: load_cylinder(directory, require_complete=True)
    elif family == 'wedge':
        directory = wedge_run({**options, 'lid_cells': 8, 'snapshot_every': .002}, root=tmp_path)
        reader = lambda: open_run(directory)
    else:
        pytest.importorskip('gmsh')
        directory = hele_run({**options, 'gap_ratio': .3, 'layer_cells': 1,
            'surface_points': 32, 'far_size': .6, 'darcy_reference': False,
            'inlet_treatment': 'energy_stable_pressure'}, root=tmp_path)
        reader = lambda: hele_load(directory)
    def forbidden():raise AssertionError('Loading an existing record constructed a PETSc solver')
    monkeypatch.setattr(multigrid, '_petsc', forbidden)
    cfg, manifest, *_ = reader()
    assert cfg['fem_load_mode'] == 'weak'
    assert cfg['mass_backend'] == mass_backend and cfg['stokes_amg_type'] == 'gamg'
    assert cfg['stokes_backend'] == 'petsc' and manifest['status'] == 'complete'
    assert manifest['source']['petsc']['communicator'] == 'COMM_SELF'
    assert manifest['source']['packages']['petsc4py']

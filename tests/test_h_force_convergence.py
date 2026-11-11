"""Analytic force samples and the new-force public run/load/analyze path."""
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.no_slip_reference_convergence.lattice import clamped_force_vector
from experiments.no_slip_reference_convergence.run import run_campaign
from experiments.plotting import latex_binary
from experiments.workflow import effective_config, load_record
from solver.mac.grid import MACGrid
from solver.mac.operators import MACOperators


def test_clamped_force_analytic_samples_constraints_and_domain():
    for nx, ny in [(32, 32), (40, 32)]:
        grid = MACGrid(nx, ny, 2.*np.pi, 2.*np.pi)
        force = clamped_force_vector(grid, 1.7)
        u, v = grid.unpack(force)
        for j in range(ny):
            for i in range(1, nx):
                expected = 1.7*(1.-math.cos(i*grid.hx))*math.sin((j+.5)*grid.hy)
                assert u[j, i] == pytest.approx(expected, abs=1e-14)
        for j in range(1, ny):
            for i in range(nx):
                expected = -1.7*math.sin((i+.5)*grid.hx)*(1.-math.cos(j*grid.hy))
                assert v[j, i] == pytest.approx(expected, abs=1e-14)
        assert np.all(u[:, [0, -1]] == 0.)
        assert np.all(v[[0, -1], :] == 0.)
        if nx == ny:
            assert np.max(np.abs(MACOperators(grid).D@force)) < 1e-13
    with pytest.raises(ValueError, match='domain'):
        clamped_force_vector(MACGrid(32, 32))
    with pytest.raises(ValueError, match='amplitude'):
        clamped_force_vector(grid, float('nan'))


def test_h_force_four_method_campaign_persists_and_reports(tmp_path):
    production = json.loads(Path('experiments/no_slip_reference_convergence/configs/h_force_256_gamma1000.json').read_text())
    config = {**production, 'base': {**production['base'], 'nx': 32, 'ny': 32,
                                   'T': .02, 'snapshots': [.01, .02], 'log_every': 10},
              'tau_base': .01, 'k_levels': [0, 1], 'reference_k': 3}
    batch = json.loads(run_campaign(config, root=tmp_path).read_text())
    assert len(batch['members']) == 8
    assert {member['scheme'] for member in batch['members']} == set(production['schemes'])
    for path in [member['path'] for member in batch['members']]+list(batch['references'].values()):
        directory = Path(path)
        cfg, _ = load_record(directory, require_complete=True)
        assert cfg['force_profile'] == 'clamped_streamfunction'
        assert cfg['force'] == 'amplitude*[(1-cos(x))*sin(y), -sin(x)*(1-cos(y))]'
        assert cfg['amplitude'] == 1. and cfg['gamma'] == 1000.
        with h5py.File(directory/'results.h5') as data:
            np.testing.assert_allclose(data['snapshots/actual_times'][:], [.01, .02], rtol=0, atol=1e-14)
            assert np.max(data['stages/divergence_inf'][:]) < 1e-10
    report = Path(batch['report'])
    analysis = json.loads((report/'analysis.json').read_text())
    if latex_binary('pdflatex') is None:
        assert analysis['pdf_status'].startswith('tool_missing')
    else:
        assert (report/'convergence.pdf').exists()
    for pair in ('sdirk2', 'sdirk3'):
        for norm in ('L2', 'H1'):
            assert (report/f'tables/{pair}_{norm}.tex').exists()
    with (report/'tables/errors.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert all(np.isfinite(float(row['L2'])) for row in rows)
    old_force = effective_config({**config['base'], 'force_profile': 'sin_x'})
    assert old_force['force'] != cfg['force']
    with pytest.raises(ValueError, match='domain'):
        effective_config({**config['base'], 'ic_kind': 'zero', 'lx': 1.})

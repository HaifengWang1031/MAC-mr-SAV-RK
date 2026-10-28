"""Independent formula and MAC-constraint checks for the signed lattice initial data."""
import math
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.no_slip_reference_convergence.lattice import force_vector, initial_velocity, streamfunction
from solver.mac.grid import MACGrid
from solver.mac.operators import MACOperators
from experiments.no_slip_reference_convergence.run import run_campaign
from experiments.workflow import load_record


def test_lattice_streamfunction_matches_literal_signed_sum():
    x = np.array([0., .31, 1.2, 2.*np.pi])
    y = np.array([0., .72, 2.1, 2.*np.pi])
    expected = np.zeros((len(y), len(x)))
    count = 0
    for k1 in range(-10, 11):
        for k2 in range(-10, 11):
            squared = k1*k1+k2*k2
            if not 0 < squared <= 100:
                continue
            count += 1
            for j, yy in enumerate(y):
                for i, xx in enumerate(x):
                    expected[j, i] += 4.*math.sin(xx/2.)**2*math.sin(yy/2.)**2 \
                        *(math.cos(k1*xx)+math.sin(k1*xx)) \
                        *(math.cos(k2*yy)+math.sin(k2*yy))/squared**1.5
    assert count == 316
    np.testing.assert_allclose(streamfunction(x, y), expected, rtol=2e-13, atol=2e-14)


def test_lattice_mac_curl_walls_divergence_amplitude_and_force():
    grid = MACGrid(40, 32, 2.*np.pi, 2.*np.pi)
    velocity = initial_velocity(grid)
    u, v = grid.unpack(velocity)
    assert np.max(np.abs(MACOperators(grid).D@velocity)) < 1e-12
    assert np.all(u[:, [0, -1]] == 0.) and np.all(v[[0, -1], :] == 0.)
    np.testing.assert_allclose(initial_velocity(grid, amplitude=8.), 2.*velocity, rtol=0., atol=0.)
    force = force_vector(grid)
    assert np.all(force[:grid.nu] == 0.)
    np.testing.assert_allclose(force[grid.nu:], np.tile(np.sin((np.arange(grid.nx)+.5)*grid.hx), grid.ny-1))
    with pytest.raises(ValueError, match='domain'):
        initial_velocity(MACGrid(40, 32))


def test_third_order_lattice_campaign_persists_and_reports(tmp_path):
    production = json.loads(Path('experiments/no_slip_reference_convergence/configs/sdirk3_256_gamma1000.json').read_text())
    config = {**production, 'base': {**production['base'], 'nx': 32, 'ny': 32,
                                   'T': .02, 'snapshots': [.01, .02], 'log_every': 10},
              'tau_base': .01, 'k_levels': [0, 1], 'reference_k': 3}
    batch = json.loads(run_campaign(config, root=tmp_path).read_text())
    assert len(batch['members']) == 4
    assert {member['scheme'] for member in batch['members']} == {'sdirk3', 'sdirk3_mrsav'}
    assert all(member['status'] == 'complete' for member in batch['members'])
    grid = MACGrid(32, 32, 2.*np.pi, 2.*np.pi)
    initial_kinetic = .5*grid.norm(initial_velocity(grid))**2
    for member in batch['members']:
        directory = Path(member['path'])
        cfg, _ = load_record(directory, require_complete=True)
        assert cfg['force'] == '[0, amplitude*sin(x)]'
        assert cfg['ic_lattice_amplitude'] == 4. and cfg['gamma'] == 1000.
        assert 'ic_seed' not in cfg and 'ic_target' not in cfg
        with h5py.File(directory/'results.h5') as data:
            np.testing.assert_allclose(data['snapshots/actual_times'][:], [.01, .02], rtol=0, atol=1e-14)
            assert data['roots/count'].shape[1] == 4
            assert np.max(data['stages/divergence_inf'][:]) < 1e-10
            assert data['diagnostics/kinetic'][0] == pytest.approx(initial_kinetic)
    report = Path(batch['report'])
    assert (report/'convergence.pdf').exists()
    assert not (report/'tables/sdirk2_L2.tex').exists()
    assert 'SDIRK3-mr-ccSAV' in (report/'tables/sdirk3_L2.tex').read_text()
    rows = list(csv.DictReader((report/'tables/errors.csv').open()))
    assert len(rows) == 8 and all(np.isfinite(float(row['L2'])) for row in rows)

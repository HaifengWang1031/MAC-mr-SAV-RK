"""Short real rerun, saved frozen stages and independently reconstructed exact fields."""
import json
from pathlib import Path

import pytest

import numpy as np

from experiments.records import digest
from tools.rerun_low_high_stability_probe import baseline_config, run

# The archived baseline record this probe replays is external, untracked evidence.
_BASELINE = (Path(__file__).resolve().parents[1]/'runs/low_high_stability_probe'
             /'20261004T080538558295'/'metadata.json')
pytestmark = pytest.mark.skipif(
    not _BASELINE.is_file(),
    reason=f'missing archived baseline record {_BASELINE.relative_to(Path(__file__).resolve().parents[1])}')


def test_current_probe_save_reload_and_exact_final_errors(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Current probe must use fixed corrections, not polynomial roots')
    monkeypatch.setattr(np, 'roots', forbidden)
    cfg = {**baseline_config(), 'nx': 12, 'T': .02}
    folder = run(cfg, tmp_path)
    index = json.loads((folder/'archive_manifest.json').read_text())
    assert all(digest(folder/name) == expected for name, expected in index['sha256'].items())
    assert json.loads((folder/'config.json').read_text()) == cfg
    n = cfg['nx']
    x, y = np.meshgrid(np.linspace(0., 1., n+1), np.linspace(0., 1., n+1))
    low_psi = np.sin(np.pi*x)**2*np.sin(np.pi*y)**2
    modes = []
    for psi in (low_psi, low_psi*np.sin(6*np.pi*x)*np.sin(6*np.pi*y)):
        psi[[0, -1], :] = 0.
        psi[:, [0, -1]] = 0.
        u, v = np.diff(psi, axis=0)*n, -np.diff(psi, axis=1)*n
        vec = np.r_[u[:, 1:-1].ravel(), v[1:-1, :].ravel()]
        modes.append(vec/np.sqrt(np.dot(vec, vec)/n**2))
    exact = cfg['low']*modes[0]+cfg['epsilon']*np.exp(-cfg['beta']*cfg['T'])*modes[1]
    initial = cfg['low']*modes[0]+cfg['epsilon']*modes[1]
    for member in index['members']:
        summary = json.loads((folder/f'{member}.json').read_text())
        assert summary['status'] == 'complete' and summary['t_final'] == cfg['T']
        with np.load(folder/f'{member}.npz') as data:
            h, stages, final = data['history'], data['stage_records'], data['final_velocity']
            np.testing.assert_array_equal(data['stage_times'], h[1:, 0])
            assert stages.shape == (len(h)-1, 4, 11)
            np.testing.assert_allclose(h[-1, 1], np.linalg.norm(final-exact)/np.linalg.norm(exact), atol=1e-13)
            np.testing.assert_allclose(h[-1, 2], np.linalg.norm(final)/np.linalg.norm(initial), atol=1e-13)
            np.testing.assert_allclose(h[1:, 5], np.max(np.abs(stages[:, :, 5]-1), axis=1))
            assert h[:, 6].max() < 1e-10 and h[:, 7:9].max() < 1e-10
            if summary['scheme'] == 'sdirk3_mrsav':
                assert summary['scalar_solver'] == 'fixed_count_frozen_factor'
                assert np.all(stages[:, :, 10] == 2.)
                np.testing.assert_array_equal(stages[:, :, 3], stages[:, :, 0])
                np.testing.assert_array_equal(stages[:, 0, 1], h[:-1, 3])
                np.testing.assert_array_equal(stages[:, 1:, 1], stages[:, :-1, 0])
                np.testing.assert_allclose(stages[:, :, 4], 1+stages[:, :, 2]+stages[:, :, 2]**2)
                np.testing.assert_allclose(stages[:, :, 5], stages[:, :, 4]*(1-stages[:, :, 0]))
            else:
                assert np.all(stages[:, :, 10] == 0.)
                assert np.isnan(stages[:, :, 1:5]).all()

"""A new tolerance, immutable parent controls and checkpoint reference queries."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.workflow import digest
from experiments.kolmogorov_adaptive.analyze import read_member
from experiments.kolmogorov_adaptive.analyze_adaptive_fixed import analyze
from experiments.kolmogorov_adaptive.run_adaptive_fixed import run as run_parent
from experiments.kolmogorov_adaptive.rerun_adaptive_fixed import run
from experiments.kolmogorov_adaptive.search_joint_tolerance import run_reference
from solver.mac.grid import MACGrid


def test_new_rtol_preserves_parent_and_matches_fresh_reference_at_new_times(tmp_path):
    spec = json.loads(Path('experiments/kolmogorov_adaptive/configs/adaptive_fixed_256.json').read_text())
    spec['common'].update(nx=8, ny=8, initial_modes=2, T=.1, output_every=.05, initial_step=.02)
    spec['repeats'] = 1
    spec['fixed_steps'] = [.1, .05]
    path = tmp_path/'old.json'
    path.write_text(json.dumps(spec))
    parent_path = run_parent(path, root=tmp_path)
    parent = json.loads(parent_path.read_text())
    parent_report = analyze(parent_path, root=tmp_path)
    protected = {p: digest(p) for directory in (parent_path.parent, parent_report)
                 for p in directory.rglob('*') if p.is_file()}
    new_spec = {**spec, 'common': {**spec['common'], 'rtol_velocity': 1e-6}}
    new_path = tmp_path/'new.json'
    new_path.write_text(json.dumps(new_spec))
    campaign_path = run(parent_path, new_path, root=tmp_path)
    campaign = json.loads(campaign_path.read_text())
    assert campaign['status'] == 'complete'
    assert campaign['config']['rtol_velocity'] == 1e-6
    assert campaign['config']['atol_velocity'] == 1e-8
    assert campaign['members'][:2] == parent['members'][1:]
    adaptive = campaign['members'][2]
    actual, _, values = read_member(Path(adaptive['path']))
    assert actual['rtol_velocity'] == 1e-6
    assert adaptive['accepted_steps'] > parent['members'][0]['accepted_steps']
    np.testing.assert_allclose(values['output_times'][-1], .1, atol=1e-12)
    grid = MACGrid(8, 8, 2*np.pi, 2*np.pi)
    for index, reference in enumerate(campaign['references']):
        assert reference['copied_queries'] >= 3
        assert reference['new_queries'] > 0
        _, _, branch = read_member(Path(reference['path']))
        fresh = run_reference(campaign['config'], reference['step'], tmp_path/f'fresh-{index}',
                              campaign['source'], lambda message: None,
                              query_times=campaign['reference_query_times'])
        _, _, full = read_member(Path(fresh['path']))
        np.testing.assert_array_equal(branch['output_times'], full['output_times'])
        for u, v, expected_u, expected_v in zip(branch['output_u'], branch['output_v'],
                                               full['output_u'], full['output_v'], strict=True):
            assert grid.norm(grid.pack(u, v)-grid.pack(expected_u, expected_v)) < 1e-9
    report = analyze(campaign_path, root=tmp_path)
    assert len(json.loads((report/'analysis.json').read_text())['summary']) == 3
    assert all(digest(p) == checksum for p, checksum in protected.items())
    invalid = {**new_spec, 'common': {**new_spec['common'], 'gamma': 2000}}
    bad_path = tmp_path/'bad.json'
    bad_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='only changed rtol_velocity'):
        run(parent_path, bad_path, root=tmp_path)


def test_production_rtol_configuration_is_separate_from_original():
    base = Path('experiments/kolmogorov_adaptive/configs')
    original = json.loads((base/'adaptive_fixed_256.json').read_text())
    new = json.loads((base/'adaptive_fixed_256_rtol_1e-6.json').read_text())
    assert original['common']['rtol_velocity'] == 2e-6
    assert new['common']['rtol_velocity'] == 1e-6
    assert {k: v for k, v in new['common'].items() if k != 'rtol_velocity'} == {
        k: v for k, v in original['common'].items() if k != 'rtol_velocity'}
    assert new['repeats'] == original['repeats'] == 3

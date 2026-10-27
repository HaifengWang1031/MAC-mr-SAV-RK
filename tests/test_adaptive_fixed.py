"""Natural adaptive outputs, shared references and six-panel persisted workflow."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.kolmogorov_adaptive.run_adaptive_fixed import reference_queries, resolved_config, run
from experiments.kolmogorov_adaptive.analyze_adaptive_fixed import (
    accepted_embedded_difference, analyze, rejected_embedded_difference,
    relative_embedded_difference, sample_indices)
from experiments.kolmogorov_adaptive.extend_adaptive_fixed import extend
from experiments.kolmogorov_adaptive.replay_rejected import replay
from experiments.kolmogorov_adaptive.run import model_and_initial, make_scheme


def test_recover_accepted_l2_difference_with_rejected_trial():
    # Accepted velocity norms 5, 10, 0 (e.g. vectors (3,4), (6,8), (0,0)).
    # Raw embedded differences .004 and .008, scale .002+.003*10=.032.
    data = {'times': np.asarray([0., .1, .2]),
            'diagnostic_kinetic': np.asarray([12.5, 50., 0.]),
            'attempt_accepted': np.asarray([False, True, True]),
            'attempt_error_u': np.asarray([100., .125, .25]),
            'attempt_t': np.asarray([0., 0., .1]),
            'attempt_h': np.asarray([.2, .1, .1])}
    embedded, scale = accepted_embedded_difference(data, atol=.002, rtol=.003)
    np.testing.assert_allclose(embedded, [.004, .008], atol=1e-16, rtol=0)
    np.testing.assert_allclose(scale, [.032, .032], atol=1e-16, rtol=0)
    np.testing.assert_allclose(relative_embedded_difference(embedded, np.asarray([10., 20.])),
                               [.0004, .0004], atol=1e-16, rtol=0)
    with pytest.raises(ValueError, match='not aligned'):
        accepted_embedded_difference({**data, 'diagnostic_kinetic': np.asarray([1.])},
                                     atol=.002, rtol=.003)


def test_unbounded_configuration_and_actual_time_reference_matching():
    _, config = resolved_config(Path('experiments/kolmogorov_adaptive/configs/adaptive_fixed_256.json'))
    assert (config['nx'], config['ny']) == (256, 256)
    assert config['max_step'] is None
    assert config['min_step'] == 1e-12
    queries = reference_queries([.100001, .10000000000001], {'T': .2, 'output_every': .1})
    np.testing.assert_allclose(queries, [0., .1, .100001, .2], atol=1e-14)
    np.testing.assert_array_equal(sample_indices(np.asarray([.100001, .2]), np.asarray(queries)), [2, 3])
    with pytest.raises(ValueError, match='do not match'):
        sample_indices(np.asarray([.15]), np.asarray(queries))


def test_replay_rejected_checkpoint_preserves_original_and_checks_raw_difference(tmp_path):
    spec = json.loads(Path('experiments/kolmogorov_adaptive/configs/adaptive_fixed_256.json').read_text())
    spec['common'].update(nx=8, ny=8, initial_modes=2, T=.1, output_every=.1, initial_step=.02)
    spec['repeats'] = 1
    spec['fixed_steps'] = [.1]
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    campaign_path = run(path, root=tmp_path)
    campaign = json.loads(campaign_path.read_text())
    member = Path(campaign['members'][0]['path'])
    before = (member/'results.npz').read_bytes()
    result = replay(member, root=tmp_path)
    metadata = json.loads((result/'manifest.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['rejected_trials'] > 0
    assert (member/'results.npz').read_bytes() == before
    with np.load(member/'results.npz') as archive:
        data = {key: archive[key] for key in archive.files}
    values = rejected_embedded_difference(result, member, data)
    config = json.loads((member/'config.json').read_text())
    model, initial = model_and_initial(config)
    first = int(values['attempt_index'][0])
    assert first == 0
    trial = make_scheme(config['scheme'], config['gamma']).step(model, initial, float(data['attempt_h'][first]))
    delta = model.vector(trial.state)-trial.embedded_velocity
    expected = np.sqrt(model.grid.area*np.sum(delta**2))
    assert values['velocity_l2_embedded_difference'][0] == pytest.approx(expected, rel=1e-12)
    high = model.vector(trial.state)
    trial_norm = np.sqrt(model.grid.area*np.sum(high**2))
    assert values['trial_velocity_l2'][0] == pytest.approx(trial_norm, rel=1e-12)
    report = analyze(campaign_path, root=tmp_path, rejected_replay=result)
    plotted = json.loads((report/'analysis.json').read_text())
    assert plotted['plot_options']['panel_e_sampling'] == 'accepted and rejected adaptive trials'
    assert plotted['plot_options']['panel_e_quantity'] == 'relative velocity L2 embedded difference'
    with np.load(report/'rejected-relative_embedded_difference.npz') as relative:
        assert relative['relative_velocity_l2_embedded_difference'][0] == pytest.approx(expected/trial_norm, rel=1e-12)
    assert (report/'rejected_difference.npz').read_bytes() == (result/'rejected_difference.npz').read_bytes()
    assert (member/'results.npz').read_bytes() == before
    bad = {**metadata, 'result_sha256': 'corrupted'}
    (result/'manifest.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='Incompatible'):
        rejected_embedded_difference(result, member, data)


def test_compute_save_reference_and_six_panel_analysis(tmp_path):
    spec = json.loads(Path('experiments/kolmogorov_adaptive/configs/adaptive_fixed_256.json').read_text())
    spec['common'].update(nx=8, ny=8, initial_modes=2, T=.1, output_every=.025)
    spec['repeats'] = 1
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    campaign_path = run(path, root=tmp_path)
    before = campaign_path.read_bytes()
    campaign = json.loads(before)
    assert len(campaign['members']) == 1+len(spec['fixed_steps'])
    assert len(campaign['references']) == 2
    adaptive = campaign['members'][0]
    with np.load(Path(adaptive['path'])/'results.npz') as values:
        assert not np.any(values['attempt_output_truncated'])
        accepted = values['attempt_accepted']
        assert np.max(values['attempt_error'][accepted]) <= 1
        assert np.any(abs(values['output_times']/.025-np.round(values['output_times']/.025)) > 1e-10)
    for member in campaign['members']:
        with np.load(Path(member['path'])/'stage_checks.npz') as checks:
            assert checks['residual'].shape == (member['accepted_steps'], 4)
            assert checks['candidates'].shape == (member['accepted_steps'], 4, 5)
            assert float(checks['final_t']) == member['final_time']
        assert member['max_stage_divergence_inf'] < 1e-8
    report = analyze(campaign_path, root=tmp_path)
    analysis = json.loads((report/'analysis.json').read_text())
    assert analysis['status'] in ('complete', 'complete_with_unverified_reference')
    assert len(analysis['summary']) == 1+len(spec['fixed_steps'])
    assert (report/'six_panel.png').stat().st_size > 1000
    assert (report/'six_panel.pdf').stat().st_size > 1000
    with np.load(report/'adaptive-embedded_difference.npz') as embedded:
        assert len(embedded['times']) == adaptive['accepted_steps']
        assert np.isfinite(embedded['velocity_l2_embedded_difference']).all()
    assert analysis['plot_options']['panel_e_sampling'] == 'accepted adaptive steps only'
    assert campaign_path.read_bytes() == before
    # A preview must use the available coarse fields, never imply refinement passed.
    partial = {**campaign, 'status': 'running', 'references': campaign['references'][:1]}
    partial_path = tmp_path/'partial.json'
    partial_path.write_text(json.dumps(partial))
    preview = analyze(partial_path, root=tmp_path, preliminary_coarse=True)
    preview_metadata = json.loads((preview/'analysis.json').read_text())
    assert preview_metadata['status'] == 'preliminary_reference_unverified'
    assert preview_metadata['error_reference_step'] == spec['reference_steps'][0]
    assert preview_metadata['max_reference_refinement_difference'] is None
    assert not any(row['reference_verified'] for row in preview_metadata['summary'])
    with np.load(preview/'adaptive-repeat-0-errors.npz') as errors:
        assert np.isnan(errors['relative_reference_difference']).all()
    assert campaign_path.read_bytes() == before


def test_extend_reuses_immutable_members_references_and_plots_nine_curves(tmp_path):
    spec = json.loads(Path('experiments/kolmogorov_adaptive/configs/adaptive_fixed_256.json').read_text())
    spec['common'].update(nx=8, ny=8, initial_modes=2, T=.1)
    spec['repeats'] = 1
    spec['fixed_steps'] = [.1, .05, .025, .001, .0005]
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    parent_path = run(path, root=tmp_path)
    before = parent_path.read_bytes()
    parent = json.loads(before)
    with pytest.raises(ValueError, match='absent'):
        extend(parent_path, [.001], root=tmp_path)
    with pytest.raises(ValueError, match='divide output_every'):
        extend(parent_path, [.003], root=tmp_path)
    campaign_path = extend(parent_path, [.01, .005, .0025], root=tmp_path)
    campaign = json.loads(campaign_path.read_text())
    assert campaign['status'] == 'complete'
    assert campaign['references'] == parent['references']
    assert campaign['members'][:6] == parent['members']
    assert len(campaign['members']) == 9
    assert [m['accepted_steps'] for m in campaign['members'][6:]] == [10, 20, 40]
    assert parent_path.read_bytes() == before
    report = analyze(campaign_path, root=tmp_path)
    assert len(json.loads((report/'analysis.json').read_text())['summary']) == 9
    bad = {**parent, 'source': {**parent['source'], 'code_sha256': 'mismatch'}}
    bad_path = tmp_path/'incompatible.json'
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='identical numerical source'):
        extend(bad_path, [.01], root=tmp_path)

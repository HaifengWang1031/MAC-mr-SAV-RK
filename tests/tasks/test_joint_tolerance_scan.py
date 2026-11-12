"""Public compute/save/read/analyze path and three-objective parameter ranking."""
import json
from pathlib import Path

import numpy as np

from experiments.kolmogorov_adaptive.analyze_joint_tolerance import analyze_scan, pareto_rows, summarize
from experiments.kolmogorov_adaptive.search_joint_tolerance import run_scan


def test_joint_scan_roundtrip_and_read_only_analysis(tmp_path):
    spec = json.loads(Path('experiments/kolmogorov_adaptive/configs/joint_tolerance_scan.json').read_text())
    spec.update(kind='pilot', repeats=1, rtol_u_values=[1e-3], atol_r_values=[1e-4, 1e-3],
                atol_u_sensitivity_values=[], velocity_only_baseline={'atol_u': 1e-8, 'rtol_u': 1e-3})
    spec['common'].update(nx=8, ny=8, T=.02, output_every=.01, initial_modes=2)
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    campaign_path = run_scan(path, root=tmp_path)
    campaign = json.loads(campaign_path.read_text())
    assert campaign['status'] == 'complete'
    assert len(campaign['members']) == 3
    assert len(campaign['references']) == 2
    before = campaign_path.read_bytes()
    for member in campaign['members']:
        directory = Path(member['path'])
        config = json.loads((directory/'config.json').read_text())
        with np.load(directory/'results.npz') as data:
            np.testing.assert_allclose(data['output_times'], [0., .01, .02], atol=1e-14)
            if config['atol_scalar'] is not None:
                accepted = data['attempt_accepted']
                assert np.max(data['attempt_error'][accepted]) <= 1.
                np.testing.assert_allclose(data['attempt_error'], np.maximum(
                    data['attempt_error_u'], data['attempt_error_r']))
                np.testing.assert_allclose(data['attempt_error_r'],
                    abs(data['attempt_scalar_defect'])/config['atol_scalar'])
        assert member['max_stage_divergence_inf'] < 1e-8
        assert member['max_stage_residual'] < 1e-8
    report = analyze_scan(campaign_path, root=tmp_path)
    analysis = json.loads((report/'analysis.json').read_text())
    assert analysis['status'] == 'complete'
    assert analysis['kind'] == 'pilot'
    assert len(analysis['summary']) == 3
    assert (report/'accuracy_cost.png').stat().st_size > 1000
    assert campaign_path.read_bytes() == before


def test_pareto_and_repeat_gate_do_not_pick_a_fast_inaccurate_or_partial_case():
    cases = [{'id': str(i), 'atol_u': 1e-8, 'rtol_u': 10.**(-i), 'atol_r': 1e-4}
             for i in range(4)]
    records = []
    for i, (cpu, error, scalar) in enumerate([(1., 1e-3, 1e-3), (2., 1e-5, 1e-4),
                                             (3., 1e-5, 1e-4), (.5, 1e-6, 1e-5)]):
        for repeat in range(1 if i == 3 else 3):
            records.append({'case_id': str(i), 'repeat': repeat, 'status': 'complete',
                            'reference_verified': True, 'integration_cpu_seconds': cpu,
                            'max_relative_velocity_error': error, 'final_relative_velocity_error': error,
                            'max_abs_r': scalar, 'accepted_steps': 10, 'rejected_trials': 0,
                            'max_step_fraction': .5})
    rows = summarize(records, cases, 3)
    assert not rows[3]['eligible']
    assert {row['id'] for row in pareto_rows(rows)} == {'0', '1'}

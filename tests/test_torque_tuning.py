"""Independent gates and small-grid run/load/reference lifecycle for tuning."""
import json
import shutil

import numpy as np
import pytest

from experiments.records import digest, write_json
from experiments.torque_transition.tune import candidates, make_plan, run_job
from experiments.torque_transition.tuning_report import analyze, qualification

GATE = {'velocity_relative': 1e-3, 'observables_relative': .01,
        'reference_fraction': .05, 'coupling': 1e-3, 'full_divergence': 1e-8}


def test_qualification_does_not_trade_accuracy_or_reference_for_speed():
    candidate = {'status': 'complete', 'max_coupling_deviation': 1e-8,
                 'max_full_divergence': 1e-12, 'max_stage_residual': 1e-13,
                 'max_error_u': .8, 'max_error_r': .5}
    assert qualification(candidate, 4e-4, 1e-6, {'energy': .001}, {'energy': 1e-6}, GATE)[0]
    assert 'velocity_budget' in qualification(candidate, .002, 1e-6, {}, {}, GATE)[1]
    assert 'reference_unresolved' in qualification(candidate, 1e-6, 1e-6, {}, {}, GATE)[1]
    candidate['max_coupling_deviation'] = .01
    assert 'sav_coupling' in qualification(candidate, 4e-4, 1e-6, {}, {}, GATE)[1]


def test_small_tuning_plan_runs_fixed_references_and_readonly_analysis(tmp_path):
    seed = tmp_path/'seed.npz'
    np.savez(seed, velocity=np.zeros(2*8*7), r=0., t=1., nodes=8, case=4, discretization='mac')
    batch = tmp_path/'batch'
    make_plan(seed, batch, duration=.02, reference_step=.002,
              settings=[candidates()[0]])
    plan = json.loads((batch/'plan.json').read_text())
    for job in plan['jobs']:
        row = run_job(job, batch/'jobs'/job['name'])
        assert row['status'] == 'complete'
        assert row['max_full_divergence'] < 1e-10
    write_json(batch/'execution.json', {'status': 'complete', 'workers': 1,
                                        'plan_sha256': digest(batch/'plan.json')})
    report = analyze(batch, tmp_path/'report')
    assert len(report['rows']) == 1
    assert report['production_T'] == 100000. and not report['production_approved']
    assert report['rows'][0]['velocity_error'] < 1e-3
    assert report['reference_refinement'] < 1e-5
    # An incomplete accepted prefix must remain visible, never get a wrong-time error or win.
    failed_batch = tmp_path/'failed-batch'
    shutil.copytree(batch, failed_batch)
    member = failed_batch/'jobs/baseline-r0'
    with np.load(member/'final.npz') as saved:
        fields = {key: saved[key].copy() for key in saved.files}
    fields['t'] = np.asarray(1.01)
    np.savez(member/'final.npz', **fields)
    failed = json.loads((member/'result.json').read_text())
    failed.update(status='failed', final_t=1.01, scalar_dominant_fraction=None)
    failed['artifacts']['final.npz'] = digest(member/'final.npz')
    write_json(member/'result.json', failed)
    failure_report = analyze(failed_batch, tmp_path/'failed-report')
    assert failure_report['winner'] is None
    assert failure_report['rows'][0]['reasons'] == ['incomplete']
    assert failure_report['rows'][0]['velocity_error'] is None
    reference_record = batch/'jobs/reference/result.json'
    original = json.loads(reference_record.read_text())
    write_json(reference_record, {**original, 'duration': .03})
    with pytest.raises(ValueError, match='reference windows'):
        analyze(batch, tmp_path/'mismatched-reference')
    write_json(reference_record, original)
    # Corrupt input must not produce a new ranking.
    (batch/'jobs/reference-half/final.npz').write_bytes(b'invalid')
    with pytest.raises(ValueError, match='checksum'):
        analyze(batch, tmp_path/'bad-report')


def test_tuning_plan_cannot_schedule_production_or_unbounded_references(tmp_path):
    with pytest.raises(ValueError, match='bounded'):
        make_plan(tmp_path/'missing.npz', tmp_path/'production', duration=100000.)
    with pytest.raises(ValueError, match='bounded'):
        make_plan(tmp_path/'missing.npz', tmp_path/'too-many-steps', reference_step=1e-12)
    assert not (tmp_path/'production').exists()

"""The benchmark replays one saved mesh/state and analyzes saved fields only."""
import json
from pathlib import Path
import shutil

from experiments.fem_scalability.report import analyze
from experiments.fem_scalability.run import PROJECT, invoke, write


def test_saved_fixture_worker_comparison_and_analysis_without_recomputation(tmp_path, monkeypatch):
    worker = PROJECT/'experiments'/'fem_scalability'
    shutil.copytree(worker, tmp_path/'worker', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('candidate_source', 'baseline_source'):
        (tmp_path/name).symlink_to(PROJECT, target_is_directory=True)
    fixture = tmp_path/'fixtures'/'cavity_0'
    fixture.mkdir(parents=True)
    write(fixture/'spec.json', dict(family='cavity', level=0, n=4))
    prepared = invoke(tmp_path, fixture, fixture/'preparation.json', 'baseline_source', 'prepare', {}, 'sdirk3_mrsav')
    assert prepared['status'] == 'complete' and prepared['velocity_dofs'] == 98
    for workload in ('replay', 'adaptive'):
        for variant, source, options in [
            ('original_recycled', 'baseline_source', dict(stokes_backend='recycled')),
            ('weak_recycled', 'candidate_source', dict(stokes_backend='recycled', fem_load_mode='weak'))]:
            output = tmp_path/'samples'/'cavity_0'/workload/'sdirk3_mrsav'/variant/'0.json'
            result = invoke(tmp_path, fixture, output, source, workload, options, 'sdirk3_mrsav')
            assert result['status'] == 'complete'
            assert len(result['times']) == 13 if workload=='replay' else len(result['times']) >= 2
            assert result['end_to_end_seconds'] >= result['online_seconds']
            if workload=='adaptive':
                assert result['nonfinite_attempt_diagnostics']
                assert all(row['accepted'] for row in result['attempts'])
    write(tmp_path/'campaign.json', dict(candidate_hashes={}, limits=['test fixture only']))
    from solver.fem.cavity import ClosedCavityModel
    def forbidden(*args, **kwargs):raise AssertionError('Analysis constructed a flow model')
    monkeypatch.setattr(ClosedCavityModel, '__init__', forbidden)
    report = analyze(tmp_path, tmp_path/'report')
    assert all(row['qualified'] for row in report['summaries'])
    weak = next(row for row in report['summaries'] if row['variant']=='weak_recycled' and row['workload']=='replay')
    assert weak['samples'][0]['errors']['velocity_inf'] < 1e-12
    assert (tmp_path/'report'/'report.md').is_file()
    assert json.loads((tmp_path/'report'/'report.json').read_text())['run'] == str(tmp_path)

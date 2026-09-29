"""No-slip Kolmogorov input identities and adaptive smoke record checks."""
import hashlib
import json
import numpy as np
import pytest

from experiments.kolmogorov_adaptive.model import force, initial_velocity, vorticity
from experiments.kolmogorov_adaptive.run import config_check, model_and_initial
from solver.mac.grid import MACGrid


def test_wall_clamped_initial_velocity_and_force_curl():
    grid = MACGrid(32,32,2*np.pi,2*np.pi)
    velocity = initial_velocity(grid,4.,3)
    from solver.mac.operators import MACOperators
    assert np.max(np.abs(MACOperators(grid).D@velocity)) < 1e-12
    u,v = grid.unpack(velocity)
    assert np.all(u[:,[0,-1]]==0.)
    assert np.all(v[[0,-1],:]==0.)
    load = force(grid,4)
    curl = vorticity(grid,load)
    _,y = grid.coordinates('vertex')
    expected = (2*np.sin(4*grid.hy/2)/grid.hy)*np.cos(4*y[1:-1,1:-1])
    np.testing.assert_allclose(curl,expected,atol=1e-13)


def test_smoke_configuration_builds_solvable_model():
    from pathlib import Path
    config = json.loads(Path('experiments/kolmogorov_adaptive/configs/smoke.json').read_text())
    config_check(config)
    model,initial = model_and_initial(config)
    assert model.grid.nx == 24
    assert initial.r == 0.
    assert np.max(np.abs(model.apply_D(model.vector(initial)))) < 1e-12


def test_adaptive_error_uses_mac_velocity_l2():
    from pathlib import Path
    from solver.adaptivity import IController, integrate_adaptive
    from solver.schemes.sdirk2 import SDIRK2
    config = json.loads(Path('experiments/kolmogorov_adaptive/configs/smoke.json').read_text())
    model,initial = model_and_initial(config)
    step = .001
    trial = SDIRK2().step(model,initial,step)
    high = model.vector(trial.state)
    low = trial.embedded_velocity
    assert low is not None
    scale = config['atol_velocity']+config['rtol_velocity']*max(
        model.grid.norm(model.vector(initial)),model.grid.norm(high))
    expected = model.grid.norm(high-low)/scale
    result = integrate_adaptive(model,SDIRK2(),initial,step,step,
                                controller=IController(atol=config['atol_velocity'],
                                                       rtol=config['rtol_velocity'],
                                                       estimator_order=2))
    np.testing.assert_allclose(result.attempts[0]['error'],expected,rtol=1e-12)


def test_fixed_schedule_keeps_nominal_step_at_output_nodes():
    from experiments.kolmogorov_adaptive.run import reference_schedule
    steps = reference_schedule([0.,.01,.02,.03,.04],.01)
    assert len(steps) == 4
    np.testing.assert_allclose(steps,[.01]*4)


def test_analysis_aligns_small_long_schedule_time_drift():
    from experiments.kolmogorov_adaptive.analyze import aligned_output_times
    config = {'T': 30., 'output_every': .1}
    expected = .1*np.arange(301)
    drifted = expected+np.linspace(0., 6e-11, len(expected))
    np.testing.assert_array_equal(
        aligned_output_times(config, {'output_times': drifted}, 'test'), expected)
    with pytest.raises(ValueError, match='incompatible'):
        aligned_output_times(
            config, {'output_times': expected+np.linspace(0., 1e-5, len(expected))},
            'test')


def test_adaptive_smoke_report_and_separate_fixed_controls(tmp_path):
    from pathlib import Path
    from experiments.kolmogorov_adaptive.run import run_campaign
    from experiments.kolmogorov_adaptive.analyze import analyze_batch
    config = json.loads(Path('experiments/kolmogorov_adaptive/configs/smoke.json').read_text())
    config.update(T=.02,initial_modes=2,reference_step=.001)
    batch_path = run_campaign(config,root=tmp_path)
    batch = json.loads(batch_path.read_text())
    assert batch['status'] == 'complete'
    assert len(batch['members']) == 8
    assert batch['fixed_members'] == []
    for item in batch['members']:
        with np.load(Path(item['path'])/'results.npz') as result:
            assert len(result['attempt_boundary_partition']) == len(result['attempt_h'])
            assert np.min(result['attempt_h']) >= config['min_step']
            assert np.isfinite(result['attempt_cpu']).all()
    report = analyze_batch(batch_path,root=tmp_path)
    assert (report/'figures/adaptive_comparison.pdf').stat().st_size > 1000
    reused_path = run_campaign(config, root=tmp_path, reuse_batch=batch_path)
    reused = json.loads(reused_path.read_text())
    assert reused['status'] == 'complete'
    assert [item['path'] for item in reused['members']] == [
        item['path'] for item in batch['members']]
    assert reused['reference'] == batch['reference']
    stale = {**batch, 'source': {**batch['source'], 'code_sha256': 'stale'}}
    stale_path = tmp_path/'stale-batch.json'
    stale_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match='different source'):
        run_campaign(config, root=tmp_path, reuse_batch=stale_path)
    from experiments.kolmogorov_adaptive.run_fixed_controls import run_fixed_controls
    from experiments.kolmogorov_adaptive.analyze_separate import analyze_separate
    comparison = run_fixed_controls(batch_path, root=tmp_path)
    comparison_batch = json.loads(comparison.read_text())
    assert comparison_batch['status'] == 'complete'
    assert len(comparison_batch['fixed_members']) == 16
    assert comparison_batch['comparison_reference_step'] == .0001
    separate_report = analyze_separate(comparison, root=tmp_path)
    assert len(list((separate_report/'figures').glob('*.pdf'))) == 8


def _synthetic_base_batch(tmp_path, *, members='complete', reference='complete'):
    """Minimal base batch for exercising the fixed-control phase gate alone."""
    from pathlib import Path
    from experiments.workflow import provenance
    config = json.loads(Path(
        'experiments/kolmogorov_adaptive/configs/smoke.json').read_text())
    source = provenance()
    member_dir = tmp_path/'member'
    reference_dir = tmp_path/'reference'
    for directory, status in ((member_dir, members), (reference_dir, reference)):
        directory.mkdir()
        (directory/'config.json').write_text(json.dumps(config))
        np.savez(directory/'results.npz', value=np.asarray([1.]))
        checksum = hashlib.sha256((directory/'results.npz').read_bytes()).hexdigest()
        (directory/'manifest.json').write_text(json.dumps({
            'status': status, 'source': source, 'result_sha256': checksum}))
    member = {'scheme': 'sdirk2', 'controller': 'I',
              'path': str(member_dir), 'status': members}
    path = tmp_path/'base.json'
    path.write_text(json.dumps({
        'status': 'complete',
        'config': config,
        'source': source,
        'members': [dict(member) for _ in range(8)],
        'fixed_members': [],
        'reference': str(reference_dir)}))
    return path


def test_fixed_control_phase_accepts_adaptive_only_base(tmp_path):
    from experiments.kolmogorov_adaptive.run_fixed_controls import _require_usable_base
    from experiments.workflow import provenance
    base = _synthetic_base_batch(tmp_path)
    _require_usable_base(json.loads(base.read_text()), provenance())


def test_fixed_control_phase_rejects_failed_adaptive_member(tmp_path):
    from experiments.kolmogorov_adaptive.run_fixed_controls import _require_usable_base
    from experiments.workflow import provenance
    base = _synthetic_base_batch(tmp_path, members='failed')
    with pytest.raises(ValueError, match='adaptive members'):
        _require_usable_base(json.loads(base.read_text()), provenance())


def test_fixed_control_phase_rejects_incomplete_reference(tmp_path):
    from experiments.kolmogorov_adaptive.run_fixed_controls import _require_usable_base
    from experiments.workflow import provenance
    base = _synthetic_base_batch(tmp_path, reference='failed')
    with pytest.raises(ValueError, match='reference'):
        _require_usable_base(json.loads(base.read_text()), provenance())


def test_tolerance_summary_selects_fastest_eligible_common_value():
    from experiments.kolmogorov_adaptive.search_tolerance import (
        CONTROLLERS, SCHEMES, summarize_records,
    )
    records = []
    for scheme in SCHEMES:
        for controller in CONTROLLERS:
            for rtol, cpu, error in ((1e-3, 1., 2e-4), (1e-4, 2., 8e-5),
                                     (5e-5, 3., 4e-5)):
                for repeat in range(3):
                    records.append({
                        'scheme': scheme, 'controller': controller, 'rtol': rtol,
                        'atol': rtol*2e-4, 'status': 'complete',
                        'integration_cpu_seconds': cpu+repeat*.01,
                        'max_relative_error': error,
                    })
    summary, recommendations, common = summarize_records(records, 1e-4)
    assert len(summary) == 24
    assert all(item['rtol'] == 1e-4 for item in recommendations)
    assert common is not None
    assert common['rtol'] == 1e-4

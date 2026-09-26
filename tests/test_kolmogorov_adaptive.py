"""No-slip Kolmogorov input identities and adaptive smoke record checks."""
import json
import numpy as np

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


def test_combined_smoke_report_contains_fixed_controls(tmp_path):
    from pathlib import Path
    from experiments.kolmogorov_adaptive.run import run_campaign
    from experiments.kolmogorov_adaptive.analyze import analyze_batch
    config = json.loads(Path('experiments/kolmogorov_adaptive/configs/smoke.json').read_text())
    config.update(T=.02,initial_modes=2,reference_step=.001)
    batch_path = run_campaign(config,root=tmp_path)
    batch = json.loads(batch_path.read_text())
    assert batch['status'] == 'complete'
    assert len(batch['members']) == 8
    assert len(batch['fixed_members']) == 4
    for item in batch['fixed_members']:
        with np.load(Path(item['path'])/'results.npz') as result:
            assert len(result['attempt_h']) == 2
            np.testing.assert_allclose(result['attempt_h'],[.01,.01])
            assert np.isfinite(result['attempt_cpu']).all()
    report = analyze_batch(batch_path,root=tmp_path)
    assert (report/'figures/adaptive_fixed_comparison.pdf').stat().st_size > 1000
    from experiments.kolmogorov_adaptive.run_fixed_controls import run_fixed_controls
    from experiments.kolmogorov_adaptive.analyze_separate import analyze_separate
    comparison = run_fixed_controls(batch_path, root=tmp_path)
    comparison_batch = json.loads(comparison.read_text())
    assert comparison_batch['status'] == 'complete'
    assert len(comparison_batch['fixed_members']) == 12
    assert comparison_batch['comparison_reference_step'] == .0001
    separate_report = analyze_separate(comparison, root=tmp_path)
    assert len(list((separate_report/'figures').glob('*.pdf'))) == 8

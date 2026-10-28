"""Exercise the public benchmark CLI at a nonzero native restart time."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.cylinder_ns.run import run
from experiments.cylinder_ns.records import fingerprints, load_reference
from experiments.cylinder_ns.campaign import cycle_sensitivity, analyze as analyze_campaign
from experiments.cylinder_ns.benchmark import REFERENCE
from experiments.cylinder_ns.analyze import analyze as analyze_cylinder

PROJECT = Path(__file__).resolve().parents[1]


def test_native_restart_worker_advances_a_duration_not_an_absolute_end(tmp_path: Path) -> None:
    seed = run({'resolution': .25, 'T': .01, 'mode': 'adaptive',
                'scheme': 'sdirk3_mrsav', 'initial_step': .001}, root=tmp_path)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLBACKEND='Agg')
    script = PROJECT/'tools/benchmark_fem_adaptive.py'
    prepared = subprocess.run([sys.executable, str(script), 'reference', '--run', str(seed),
                               '--root', str(tmp_path), '--duration', '.002', '--step', '.001',
                               '--prepare-only'], cwd=PROJECT, env=env, text=True,
                              capture_output=True, check=True, timeout=30)
    campaign = Path(prepared.stdout.strip().splitlines()[-1])
    output = campaign/'adaptive-test'
    subprocess.run([sys.executable, str(script), 'worker', '--source', str(PROJECT),
                    '--input', str(campaign/'input'), '--output', str(output),
                    '--backend', 'direct', '--duration', '.002'], cwd=PROJECT, env=env,
                   capture_output=True, text=True, check=True, timeout=30)
    result = json.loads(output.with_suffix('.json').read_text())
    assert result['status'] == 'complete'
    assert result['initial_time'] == pytest.approx(.01)
    assert result['final_time'] == pytest.approx(.012, abs=1e-14)
    with np.load(output.with_suffix('.npz'), allow_pickle=False) as fields:
        assert fields['times'][-1] == pytest.approx(.012, abs=1e-14)


@pytest.mark.parametrize('adaptive', [False, True])
def test_native_record_reads_without_a_model_and_rejects_corruption(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch,
                                                                  adaptive: bool) -> None:
    seed = run({'resolution': .25, 'T': .01, 'mode': 'adaptive', 'initial_step': .001}, root=tmp_path)
    env = os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    control = (['--adaptive', '--control-config', str(PROJECT/'experiments/cylinder_ns/configs/native_tight.json')]
               if adaptive else ['--step', '.001'])
    process = subprocess.run([sys.executable, str(PROJECT/'tools/benchmark_fem_adaptive.py'), 'reference',
                              '--run', str(seed), '--root', str(tmp_path), '--duration', '.002', *control],
                             cwd=PROJECT, env=env, text=True, capture_output=True, check=True, timeout=30)
    campaign = Path(process.stdout.strip().splitlines()[-1])
    before = fingerprints(campaign)
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError('Readonly reconstruction must not create a dynamical model')
    monkeypatch.setattr('solver.fem.open_flow.OpenFlowModel.__init__', forbidden)
    record = load_reference(campaign)
    assert record['result']['final_time'] == pytest.approx(.012, abs=1e-14)
    assert record['fields']['u'].shape == record['native']['u'].shape
    assert record['fields']['physical_pressure'].shape == (record['pressure_basis'].N,)
    assert record['fields']['scalar_history'].shape == record['fields']['times'].shape
    report = analyze_cylinder(campaign, root=tmp_path)
    assert (report/'figures/velocity.png').is_file()
    assert (report/'figures/steps.png').is_file()
    assert not json.loads((report/'analysis.json').read_text())['reference_comparison']['passed']
    assert fingerprints(campaign) == before
    path = campaign/('adaptive-direct-0.npz' if adaptive else 'fixed-direct-0.npz')
    payload = bytearray(path.read_bytes()); payload[-1] ^= 1; path.write_bytes(payload)
    with pytest.raises(ValueError, match='checksum'):
        load_reference(campaign)


def test_cycle_sensitivity_gates_primary_quantities_not_near_zero_mean_lift() -> None:
    base = {**REFERENCE, 'lift_mean': 0.}
    changed = {**base, 'lift_max': REFERENCE['lift_max']*1.01, 'lift_mean': .001}
    assert cycle_sensitivity(changed, base, tolerance=.02)['passed']
    changed['lift_max'] = REFERENCE['lift_max']*1.03
    result = cycle_sensitivity(changed, base, tolerance=.02)
    assert not result['passed']
    assert result['relative_differences']['lift_max'] == pytest.approx(.03)
    assert not cycle_sensitivity({}, base, tolerance=.02)['passed']


@pytest.mark.parametrize('adaptive', [False, True])
def test_explicit_refinement_report_is_readonly_and_does_not_qualify_a_short_run(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch,
                                                                             adaptive: bool) -> None:
    seed = run({'resolution': .25, 'T': .01, 'mode': 'adaptive'}, root=tmp_path)
    env = os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLBACKEND='Agg')
    paths = []
    for index, (resolution, segments, step) in enumerate([(.25, 24, .001), (.5, 48, .001), (.5, 48, .0005), (.5, 96, .001)]):
        control = (['--adaptive', '--control-config', str(PROJECT/'experiments/cylinder_ns/configs'/
                     ('native_tight.json' if index == 2 else 'native_adaptive.json'))]
                   if adaptive else ['--step', str(step)])
        result = subprocess.run([sys.executable, str(PROJECT/'tools/benchmark_fem_adaptive.py'), 'reference',
                                 '--run', str(seed), '--root', str(tmp_path), '--duration', '.002',
                                 '--resolution', str(resolution), '--circle-segments', str(segments), *control],
                                cwd=PROJECT, env=env, text=True, capture_output=True, check=True, timeout=30)
        paths.append(Path(result.stdout.strip().splitlines()[-1]))
    before = [fingerprints(path) for path in paths]
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError('A refinement report must not launch computation')
    monkeypatch.setattr('solver.fem.open_flow.OpenFlowModel.__init__', forbidden)
    report = analyze_campaign(*paths, root=tmp_path)
    summary = json.loads((report/'analysis.json').read_text())
    assert not summary['qualified'] and not summary['developed_all']
    assert not summary['temporal_sensitivity']['passed']
    assert (report/'figures/vorticity.png').is_file()
    assert [fingerprints(path) for path in paths] == before

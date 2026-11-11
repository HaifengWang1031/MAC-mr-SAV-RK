"""Production persistence regressions; added but not run for the no-more-tests launch."""
import csv
import json
from threading import Event
from types import SimpleNamespace

import numpy as np

from experiments.torque_transition.production import Recorder, SCALAR_KEYS
from solver.adaptivity import PIController, ScalarErrorControl, integrate_adaptive
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV
from tests.test_sdirk3 import ODEModel, State


def test_bounded_history_and_joint_pi_continuation_keep_the_natural_trajectory():
    model = ODEModel()
    initial = State(0., model.exact(0.))
    controller = PIController(atol=1e-8, rtol=1e-5, max_step=.015,
                              proportional=.8, integral=.3)
    options = {'controller': controller, 'scalar_control': ScalarErrorControl(1e-3, 1000.)}
    full = integrate_adaptive(model, SDIRK3MRSAV(1000.), initial, .4, .001, **options)

    def stop_at_natural_node(result):
        if result.accepted_steps == 5:
            raise RuntimeError('intentional checkpoint boundary')

    prefix = integrate_adaptive(model, SDIRK3MRSAV(1000.), initial, .4, .001,
                                retain_history=False, on_advance=stop_at_natural_node, **options)
    assert prefix.status == 'failed' and prefix.accepted_steps == 5
    assert prefix.continuation.previous_u is not None
    assert prefix.continuation.previous_r is not None
    tail = integrate_adaptive(model, SDIRK3MRSAV(1000.), prefix.final, .4-prefix.final.t, .001,
                              retain_history=False, continuation=prefix.continuation, **options)
    assert full.status == tail.status == 'complete'
    assert len(tail.times) == 2 and len(tail.stages) == len(tail.attempts) == 1
    assert tail.continuation.accepted_steps == full.accepted_steps
    assert tail.continuation.attempted_steps == full.attempted_steps
    np.testing.assert_allclose(tail.final.velocity, full.final.velocity, rtol=0., atol=1e-13)
    assert abs(tail.final.r-full.final.r) < 1e-13
    assert np.linalg.norm(full.final.velocity-model.exact(.4)) < 1e-3


def test_nearest_sampler_and_checkpoint_offsets_restore_without_duplicate_output(tmp_path, monkeypatch):
    def state(t):
        return SimpleNamespace(t=t, r=0., velocity=np.full(2*256*255, t))

    def observables(model, saved, case):
        return {key: saved.t if key == 'energy' else 0. for key in SCALAR_KEYS[3:]}

    monkeypatch.setattr('experiments.torque_transition.production.mac_observables', observables)
    (tmp_path/'fields').mkdir()
    model = SimpleNamespace(vector=lambda saved: saved.velocity)
    recorder = Recorder(tmp_path, model, 4, 'fixed-plan', state(0.), None, Event())
    recorder.sample(state(1.5))  # uncommitted output must disappear on restoration
    recorder.streams.close()
    with np.load(tmp_path/'checkpoint.npz') as saved:
        metadata = json.loads(str(saved['metadata']))
    resumed = Recorder(tmp_path, model, 4, 'fixed-plan', state(0.), metadata, Event())
    resumed.sample(state(1.5))
    resumed.sample(state(2.5))
    resumed.streams.close()
    with (tmp_path/'scalars.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row['request_t']) for row in rows] == [0., 1., 2.]
    # 2 is exactly midway: the earlier natural accepted node must win.
    assert [float(row['actual_t']) for row in rows] == [0., 1.5, 1.5]

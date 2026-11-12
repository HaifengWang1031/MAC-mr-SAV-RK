"""Adaptive-driver integration checks for the tensor Stokes backend."""
import json
from pathlib import Path

import numpy as np

from experiments.kolmogorov_adaptive.run import config_check, model_and_initial
from solver.adaptivity import IController, integrate_adaptive
from solver.mac.stokes import DirectStokes
from solver.mac.tensor_stokes import TensorStokes
from solver.schemes.sdirk3 import SDIRK3


def _smoke_config() -> dict:
    return json.loads(Path(
        'experiments/kolmogorov_adaptive/configs/smoke.json').read_text())


def test_adaptive_configuration_selects_tensor_and_legacy_defaults_to_direct():
    config = _smoke_config()
    config_check(config)
    model,_ = model_and_initial(config)
    assert isinstance(model.backend,TensorStokes)
    for key in ('stokes_backend','stokes_krylov_tolerance','stokes_max_iterations'):
        config.pop(key)
    config_check(config)
    legacy,_ = model_and_initial(config)
    assert isinstance(legacy.backend,DirectStokes)


def test_adaptive_tensor_backend_matches_direct_rejection_and_step_control():
    config = _smoke_config()
    config.update(nx=12,ny=10,initial_modes=2)
    results = []
    for backend in ('direct','tensor'):
        case = {**config,'stokes_backend':backend}
        model,initial = model_and_initial(case)
        controller = IController(atol=1e-12,rtol=1e-8,safety=case['safety'],
                                 min_step=1e-7,max_step=case['max_step'],estimator_order=3)
        results.append((model,integrate_adaptive(model,SDIRK3(),initial,.002,.002,
                                                 controller=controller)))
    direct_model,direct = results[0]
    tensor_model,tensor = results[1]
    assert direct.status == tensor.status == 'complete'
    assert [item['accepted'] for item in direct.attempts] == [False,True,True]
    np.testing.assert_array_equal([item['accepted'] for item in tensor.attempts],
                                  [item['accepted'] for item in direct.attempts])
    np.testing.assert_allclose([item['step'] for item in tensor.attempts],
                               [item['step'] for item in direct.attempts],rtol=2e-9)
    np.testing.assert_allclose(tensor_model.vector(tensor.final),
                               direct_model.vector(direct.final),rtol=2e-11,atol=2e-11)

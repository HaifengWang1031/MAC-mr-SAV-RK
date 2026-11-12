"""Tensor/direct adaptive equivalence from explicit no-slip mathematical fields."""
import numpy as np

from solver.adaptivity import IController, integrate_adaptive
from solver.mac.grid import MACGrid
from solver.mac.tensor_stokes import TensorStokes
from solver.mac_ns import MACNavierStokes
from solver.schemes.sdirk3 import SDIRK3
from tests.references.mac import forcing, initial_velocity


def test_tensor_backend_keeps_direct_rejections_steps_and_full_constraints() -> None:
    results = []
    for tensor in (False, True):
        grid = MACGrid(12, 10, 1.3, .8)
        model = MACNavierStokes(grid, .1)
        if tensor:
            model.backend = TensorStokes(model.ops, tolerance=1e-9, krylov_tolerance=1e-12)
        model.force = lambda t: forcing(grid, .1, .2, t)
        initial = model.state(0., initial_velocity(grid, .2))
        result = integrate_adaptive(model, SDIRK3(), initial, .002, .002,
                                    controller=IController(atol=1e-12, rtol=1e-8,
                                                           min_step=1e-7, max_step=.002))
        assert result.status == 'complete', result.error
        assert result.final.t == .002
        assert model.max_abs(model.apply_D(model.vector(result.final))) < 1e-11
        assert max(stage.continuity_residual for stage in result.final_stages) < 1e-11
        results.append((model, result))
    direct_model, direct = results[0]
    tensor_model, tensor = results[1]
    assert [a['accepted'] for a in direct.attempts] == [False, True, True]
    np.testing.assert_array_equal([a['accepted'] for a in tensor.attempts],
                                  [a['accepted'] for a in direct.attempts])
    np.testing.assert_allclose([a['step'] for a in tensor.attempts],
                               [a['step'] for a in direct.attempts], rtol=2e-8)
    np.testing.assert_allclose(tensor_model.vector(tensor.final),
                               direct_model.vector(direct.final), rtol=2e-11, atol=2e-11)

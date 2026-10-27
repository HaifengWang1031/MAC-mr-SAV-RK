"""One-stage pressure, complete constraints and fixed-lifting checks across backends."""
import numpy as np
import pytest

from solver.channel import ChannelGrid, ChannelModel
from solver.fourier import FourierModel, FourierSpace
from solver.mac.grid import MACGrid
from solver.mac.tensor_stokes import TensorStokes
from solver.mac_ns import MACNavierStokes
from solver.schemes import IMEXEuler, IMEXEulerMRSAV
from solver.spectral.assembly import Space
from solver.spectral.lifting import LidLifting
from solver.spectral.model import SpectralModel


@pytest.mark.parametrize('backend', ['mac_direct', 'mac_tensor', 'channel_direct', 'channel_fft', 'fourier', 'legendre'])
@pytest.mark.parametrize('scheme', [IMEXEuler(), IMEXEulerMRSAV(1.)])
def test_gradient_load_returns_step_pressure_and_zero_velocity(backend, scheme):
    if backend.startswith('mac'):
        model = MACNavierStokes(MACGrid(8, 6), .1)
        if backend == 'mac_tensor': model.backend = TensorStokes(model.ops)
        x, y = model.grid.coordinates('p')
        p = np.cos(np.pi*x)*np.cos(np.pi*y)
        zero = np.zeros(model.grid.size)
    elif backend.startswith('channel'):
        model = ChannelModel(ChannelGrid(8, 6), .1, backend=backend.split('_')[1])
        x, y = model.grid.coordinates('p')
        p = np.cos(x)*np.cos(np.pi*y)
        zero = np.zeros(model.grid.size)
    elif backend == 'fourier':
        model = FourierModel(FourierSpace(18, 15), .1)
        p = model.space.to_spectral(np.cos(model.space.x)*np.sin(model.space.y))
        zero = model.zero_velocity()
    else:
        model = SpectralModel(Space(6), .1)
        p = np.zeros(model.modes)
        p[1] = .3
        p[model.space.size] = -.2
        zero = model.zero_velocity()
    model.force = lambda t: model.apply_G(p)
    trial = scheme.step(model, model.state(0., zero), .03)
    stage = trial.stages[0]
    assert model.max_abs(model.vector(trial.state)) < 1e-11
    expected_pressure = model.space.to_physical(p) if backend == 'fourier' else p
    np.testing.assert_allclose(stage.pressure, expected_pressure, atol=1e-10)
    assert stage.continuity_residual < 1e-11 and stage.residual < 1e-11


@pytest.mark.parametrize('scheme', [IMEXEuler(), IMEXEulerMRSAV(1.)])
def test_fixed_lifting_obeys_momentum_scalar_pairing_and_full_constraint(scheme):
    space = Space(6)
    model = SpectralModel(space, .1, lifting=LidLifting(space))
    # Enforce the inhomogeneous weak constraint before starting the time step.
    old = model.solve(model.zero_velocity(), mass=1., viscosity=.01).velocity
    initial = model.state(0., old)
    h = .01
    trial = scheme.step(model, initial, h)
    U, r = model.vector(trial.state), trial.state.r
    B, work = model.nonlinear_with_lifting(old)
    factor = 1-r if isinstance(scheme, IMEXEulerMRSAV) else 1.
    defect = model.combine((1., U), (-1., old), (h*model.nu, model.apply_K(U)),
                           (h, model.apply_G(trial.stages[0].pressure)), (h*factor, B))
    assert model.max_abs(defect) < 1e-10
    assert model.max_abs(model.apply_D(U)) < 1e-11
    if isinstance(scheme, IMEXEulerMRSAV):
        assert abs((1+h)*r+h*(model.inner(B, U)+work)) < 1e-12


@pytest.mark.parametrize('scheme', [IMEXEuler(), IMEXEulerMRSAV()])
def test_one_stage_rejects_a_corrupt_full_divergence_even_with_zero_solve_residual(scheme, monkeypatch):
    model = MACNavierStokes(MACGrid(6, 4), .1)
    monkeypatch.setattr(model, 'stage_divergences', lambda U: (1., 1.))
    initial = model.state(0., np.zeros(model.grid.size))
    with pytest.raises(RuntimeError, match='divergence'):
        scheme.step(model, initial, .01)

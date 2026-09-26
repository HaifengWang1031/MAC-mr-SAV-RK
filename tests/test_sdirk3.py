"""Independent ODE order check from Obsidian note 15 and MAC persistence."""
from dataclasses import dataclass
import h5py
import numpy as np
import pytest
from experiments.workflow import run_experiment
from solver.core import Stage
from solver.integrate import integrate
from solver.schemes.sdirk3 import C, D, DHAT, SDIRK3
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV


@dataclass
class State:
    t: float
    velocity: np.ndarray
    r: float = 0.


@dataclass
class Solution:
    velocity: np.ndarray
    pressure: np.ndarray
    residual: float = 0.
    divergence_inf: float = 0.


class ODEModel:
    nu = 1.
    L = np.diag([.7, 1.1])

    def vector(self, state): return state.velocity
    def state(self, t, velocity, r=0.): return State(t, velocity, r)
    def combine(self, *terms): return sum((weight*value for weight,value in terms), np.zeros_like(terms[0][1]))
    def apply_K(self, velocity): return self.L@velocity
    def apply_G(self, pressure): return np.zeros(2)
    def apply_D(self, velocity): return np.zeros(1)
    def physical_divergence_inf(self, velocity): return 0.
    def inner(self, left, right): return float(left@right)
    def max_abs(self, field): return float(np.max(np.abs(field)))
    def nonlinear(self, velocity):
        x,y=velocity
        return 3*x*np.array([-y,x])
    def nonlinear_with_lifting(self, velocity): return self.nonlinear(velocity),0.
    def exact(self, t): return np.array([np.cos(t),np.sin(t)])
    def force(self, t):
        u=self.exact(t)
        return np.array([-np.sin(t),np.cos(t)])+self.L@u+self.nonlinear(u)
    def solve(self, rhs, *, mass, viscosity): return Solution(np.linalg.solve(mass*np.eye(2)+viscosity*self.L,rhs),np.zeros(2))
    def solve_columns(self, columns, *, mass, viscosity): return [self.solve(col,mass=mass,viscosity=viscosity) for col in columns]
    def stage(self, pressure, residual, divergence_inf, r=0., candidates=(), root_residuals=(),
              scalar_residual=0., *, continuity_residual=float('nan')):
        return Stage(pressure,residual,divergence_inf,r,list(candidates),list(root_residuals),
                     scalar_residual,continuity_residual)
    def diagnostics(self, state): return {'error':float(np.linalg.norm(state.velocity-self.exact(state.t)))}


def test_coefficients_and_third_order_on_independent_ode():
    assert np.allclose(D.sum(axis=1),DHAT.sum(axis=1))
    np.testing.assert_allclose(D.sum(axis=1),np.diff(C),atol=1e-15)
    model=ODEModel()
    for scheme in (SDIRK3(),SDIRK3MRSAV()):
        errors=[]
        for count in (50,100,200):
            result=integrate(model,scheme,State(0.,model.exact(0.)),[2/count]*count)
            assert result.status=='complete',result.error
            assert len(result.final_stages)==4
            errors.append(np.linalg.norm(result.final.velocity-model.exact(2.)))
        assert min(np.log2(np.array(errors[:-1])/errors[1:]))>2.85,(scheme.name,errors)


@pytest.mark.parametrize('scheme', ['sdirk3','sdirk3_mrsav'])
def test_four_stages_are_saved_and_prescribed_steps_work(tmp_path,scheme):
    path=run_experiment({'experiment':'forced_ns','scheme':scheme,'nx':8,'ny':6,
                         'T':.02,'steps':[.007,.013],'m':1},root=tmp_path)
    with h5py.File(path/'results.h5') as data:
        assert data.attrs['status']=='complete'
        assert data['stages/residual'].shape==(2,4)
        assert data['roots/candidates'].shape==(2,4,5)
        assert data['final/stage_pressure'].shape[0]==4


def test_steady_mac_velocity_and_zero_scalar_are_preserved():
    from solver.mac.grid import MACGrid
    from solver.mac_ns import MACNavierStokes
    grid = MACGrid(8, 6)
    model = MACNavierStokes(grid, .1)
    drive = np.random.default_rng(7).normal(size=grid.size)
    steady = model.backend.solve(drive,mass=1.,viscosity=.1).velocity
    force = model.nu*model.apply_K(steady)+model.nonlinear(steady)
    model.force = lambda t: force
    state = model.state(0.,steady)
    for method in (SDIRK3(),SDIRK3MRSAV()):
        trial=method.step(model,state,.013)
        assert grid.norm(model.vector(trial.state)-steady)<1e-11
        assert abs(trial.state.r)<1e-11


def test_sdirk3_campaign_uses_sdirk3_reference_and_labels(tmp_path):
    import json
    from experiments.forced_ns_convergence.run import run_campaign
    cfg={'base':{'experiment':'forced_ns','nx':8,'ny':8,'m':1,'T':.04,
                 'snapshots':[.04],'log_every':100},
         'tau_base':.08,'k_levels':[1,2],'reference_k':4,
         'sensitivity_threshold':.5,'max_reference_refinements':0,
         'schemes':['sdirk3','sdirk3_mrsav']}
    batch_path=run_campaign(cfg,root=tmp_path)
    batch=json.loads(batch_path.read_text())
    assert [entry['scheme'] for entry in batch['trials']]==['sdirk3','sdirk3_mrsav']*2
    assert batch['references']['reference_k']==4
    report=__import__('pathlib').Path(batch['reports'][0])
    latex=(report/'tables/L2.tex').read_text()
    assert 'SDIRK3-mrSAV' in latex and 'order 2' not in latex


def test_spectral_lifting_runs_all_four_stages():
    from solver.spectral.assembly import Space
    from solver.spectral.lifting import LidLifting
    from solver.spectral.model import SpectralModel
    space=Space(4,1.,1.)
    model=SpectralModel(space,.1,lifting=LidLifting(space))
    initial=model.state(0.,model.zero_velocity())
    for method in (SDIRK3(),SDIRK3MRSAV()):
        trial=method.step(model,initial,.01)
        assert len(trial.stages)==4
        assert np.isfinite(model.vector(trial.state)).all()
        assert max(stage.continuity_residual for stage in trial.stages)<1e-10

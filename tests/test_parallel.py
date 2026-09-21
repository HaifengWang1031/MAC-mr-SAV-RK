"""Public numerical seams; run explicitly under MPI, never replicate a global solve."""
import numpy as np
import pytest
pytest.importorskip("petsc4py")
from solver.mac_parallel.stokes import ParallelStokes
from solver.mac.grid import MACGrid


def test_pure_pressure_gradient_recovers_zero_velocity():
    solver=ParallelStokes(MACGrid(8,6),tolerance=1e-10)
    pressure=solver.pressure_vector(lambda x,y: x*x+y*y)
    rhs=solver.gradient(pressure)
    result=solver.solve(rhs,mass=1.,viscosity=.03)
    assert result.velocity.norm()<1e-7
    assert result.divergence_inf<1e-8
    assert result.residual<1e-8
    error=result.pressure.copy();error.axpy(-1.,pressure)
    assert error.norm()<1e-6
    solver.close()


def test_partitioned_stencils_and_stokes_match_serial_reference():
    from solver.mac.operators import MACOperators
    from solver.mac.stokes import DirectStokes
    from solver.mac.kernels import convection
    from petsc4py import PETSc
    g=MACGrid(9,7,1.3,.8)
    solver=ParallelStokes(g,tolerance=1e-11)
    field=solver.layout.vector(lambda x,y: np.sin(3*x+2*y),lambda x,y: np.cos(x-4*y))
    lap=field.duplicate();solver.K.mult(field,lap)
    nonlinear=solver.layout.nonlinear(field)
    solved=solver.solve(field,mass=1.,viscosity=.07)
    packed=solver.layout.gather(field);glap=solver.layout.gather(lap);gn=solver.layout.gather(nonlinear)
    gu=solver.layout.gather(solved.velocity);gp=solver.layout.gather_pressure(solved.pressure)
    errors=None
    if solver.comm.rank==0:
        vec=g.pack(*packed);op=MACOperators(g)
        direct=DirectStokes(op).solve(vec,mass=1.,viscosity=.07)
        cu,cv=convection(*packed,g.hx,g.hy)
        errors=[np.max(np.abs(g.pack(*glap)-op.K@vec)),np.max(np.abs(g.pack(*gn)-g.pack(cu,cv))),
                np.max(np.abs(g.pack(*gu)-direct.velocity)),np.max(np.abs(gp-direct.pressure))]
    errors=solver.comm.bcast(errors,root=0)
    assert max(errors)<1e-7,errors
    assert solver.layout.local_n<g.size or solver.comm.size==1
    for obj in (field,lap,nonlinear,solved.velocity,solved.pressure):obj.destroy()
    solver.close()


@pytest.mark.parametrize('sav',[False,True])
def test_distributed_time_steps_match_serial_sdirk(sav):
    from solver.mac_parallel.integrate import ParallelNS,ParallelSDIRK2
    from solver.mac_ns import MACNavierStokes
    from solver.schemes.sdirk2 import SDIRK2
    from solver.schemes.sdirk2_mrsav import SDIRK2MRSAV
    from experiments.problems import initial_velocity
    grid=MACGrid(12,9)
    model=ParallelNS(grid,.1,tolerance=1e-12)
    state=model.initial(amplitude=.2)
    stepper=ParallelSDIRK2(sav=sav,gamma=1.)
    roots=[]
    for dt in (.01,.006,.014):
        trial=stepper.step(model,state,dt)
        state.velocity.destroy();state=trial.state
        roots.append([s.r for s in trial.stages])
        for stage in trial.stages:stage.pressure.destroy()
    fields=model.stokes.layout.gather(state.velocity)
    error=None
    if model.comm.rank==0:
        serial=MACNavierStokes(grid,.1)
        expected=serial.state(0,initial_velocity(grid,.2))
        scheme=SDIRK2MRSAV(1.) if sav else SDIRK2()
        expected_roots=[]
        for dt in (.01,.006,.014):
            tr=scheme.step(serial,expected,dt);expected=tr.state
            expected_roots.append([s.r for s in tr.stages])
        error=max(np.max(np.abs(grid.pack(*fields)-serial.vector(expected))),abs(state.r-expected.r),
                  np.max(np.abs(np.array(roots)-expected_roots)))
    error=model.comm.bcast(error,root=0)
    assert error<1e-8,error
    assert model.diagnostics(state)['divergence_inf']<1e-8
    state.velocity.destroy();model.close()


def test_parallel_run_reuse_and_failed_prefix(tmp_path):
    import json
    from mpi4py import MPI
    from experiments.parallel_ns.run import run_parallel
    root=MPI.COMM_WORLD.bcast(str(tmp_path) if MPI.COMM_WORLD.rank==0 else None,root=0)
    from pathlib import Path
    cfg={'experiment':'decay','nx':8,'ny':7,'nu':.1,'T':.02,'steps':[.01,.01]}
    first=run_parallel(cfg,root=Path(root))
    assert run_parallel(cfg,root=Path(root))==first
    failure=run_parallel({**cfg,'max_iterations':1},root=Path(root))
    message=None
    if MPI.COMM_WORLD.rank==0:
        good=json.loads((first/'manifest.json').read_text())
        bad=json.loads((failure/'manifest.json').read_text())
        message=(good['status'],bad['status'],bad['accepted_steps'])
    assert MPI.COMM_WORLD.bcast(message,root=0)==('complete','failed',0)


def test_single_rank_petsc_adapter_matches_direct_backend():
    """The PETSc adapter and SuperLU are independent stacks solving the same operator.

    PETScStokes drives the unchanged serial scheme code through the injected backend,
    so agreement here cross-checks the distributed assembly, the block preconditioner
    and the divergence handling against SuperLU on the same small problem.
    """
    from solver.mac.operators import MACOperators
    from solver.mac.stokes import DirectStokes
    from solver.mac_ns import MACNavierStokes
    from solver.schemes.sdirk2 import SDIRK2
    from solver.mac_parallel.serial_adapter import PETScStokes
    from experiments.problems import initial_velocity
    rng=np.random.default_rng(11)
    g=MACGrid(9,7,1.3,.8)
    rhs=rng.standard_normal(g.size)
    reference=DirectStokes(MACOperators(g)).solve(rhs,mass=1.,viscosity=.07)
    adapted=PETScStokes(MACOperators(g),tolerance=1e-12).solve(rhs,mass=1.,viscosity=.07)
    assert np.max(np.abs(adapted.velocity-reference.velocity))<1e-7
    assert np.max(np.abs(adapted.pressure-reference.pressure))<1e-7
    assert adapted.divergence_inf<1e-8 and adapted.residual<1e-8
    vectors=[]
    for backend in (DirectStokes(MACOperators(g)),PETScStokes(MACOperators(g),tolerance=1e-12)):
        model=MACNavierStokes(g,.1,backend=backend)
        state=model.state(0.,initial_velocity(g,.1))
        trial=SDIRK2().step(model,state,1e-3)
        vectors.append(model.vector(trial.state))
    assert np.max(np.abs(vectors[0]-vectors[1]))<1e-7

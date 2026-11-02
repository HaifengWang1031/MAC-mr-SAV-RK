"""Replay one SAV step from an explicit failed record, never modify that record.

The pass/fail signal is growth of the observed large inlet-corner velocity spike
at a fixed 1e-7 step. This is a diagnostic state replay, not checkpoint resume.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import h5py
import numpy as np

from experiments.hele_shaw_airfoil.records import load
from experiments.workflow import PROJECT, digest, write_json
from experiments.periodic_ns.run import make_controller
from solver.adaptivity import ScalarErrorControl, integrate_adaptive
from solver.core import State
from solver.fem.hele_shaw import HeleShawModel
from solver.fem.space import MeshSpace
from solver.schemes.registry import make_scheme


def probe(directory: Path, *, inlet_treatment: str = 'natural',
          backend: str = 'direct', dt: float = 1e-7,
          replay_to: float | None = None) -> dict[str, Any]:
    cfg, manifest, mesh, _, _ = load(directory, require_complete=False)
    options = {} if inlet_treatment == 'natural' else {'inlet_treatment': inlet_treatment}
    model = HeleShawModel(MeshSpace(mesh, cfg['quadrature_order']), cfg['nu'], cfg['gap'],
        pressure_drop=cfg['pressure_drop'], ramp_time=cfg['ramp_time'], rho=cfg['rho'],
        convection=cfg['convection'], outlet_treatment=cfg['outlet_treatment'],
        tolerance=cfg['stokes_tolerance'], cache_size=1, stokes_backend=backend,
        stokes_ordering=cfg['stokes_ordering'], **options)
    with h5py.File(directory/'results.h5', 'r') as data:
        if replay_to is None:
            state = State(float(data['final/t'][()]), data['final/u'][:], data['final/v'][:],
                          float(data['final/r'][()]))
        else:
            i = int(np.argmin(abs(data['snapshots/actual_times'][:]-.7)))
            state = State(float(data['snapshots/actual_times'][i]), data['snapshots/u'][i],
                          data['snapshots/v'][i], float(data['snapshots/r'][i]))
    coordinates = model.velocity_basis.split_bases()[0].doflocs
    xmin, _, ymin, _ = np.asarray(cfg['bounds'])*cfg['chord']
    corner = (coordinates[0] < xmin+.05*cfg['chord']) & (coordinates[1] < ymin+.05*cfg['chord'])
    before = float(np.max(np.hypot(state.u[corner], state.v[corner])))
    if replay_to is not None:
        history = []
        def guard(current: State) -> None:
            peak = float(np.max(np.hypot(current.u, current.v)))
            history.append(dict(t=current.t, r=current.r, maximum_nodal_speed=peak,
                corner_speed=float(np.max(np.hypot(current.u[corner], current.v[corner])))))
            if len(history) % 10 == 0:
                print(json.dumps(history[-1]), flush=True)
            if peak > 20. or abs(current.r) > .05:
                raise RuntimeError('Replay qualification limit exceeded')
        trajectory = integrate_adaptive(model, make_scheme(cfg['scheme'], cfg['gamma']), state,
            replay_to-state.t, .001, controller=make_controller(cfg['controller'], cfg['scheme']),
            scalar_control=ScalarErrorControl(cfg['scalar_atol'], cfg['gamma']), on_accept=guard)
        return dict(source_run=str(directory.resolve()), source_results_sha256=manifest['results_sha256'],
            inlet_treatment=inlet_treatment, backend=backend, initial_time=state.t,
            target_time=replay_to, final_time=trajectory.final.t, status=trajectory.status,
            error=trajectory.error, accepted_steps=len(trajectory.times)-1, history=history,
            passed=trajectory.status == 'complete' and trajectory.final.t == replay_to,
            qualification='Controlled state replay, with PI history reset; not checkpoint continuation or full production.')
    assert before > 100., 'The captured trace must contain the actual large corner spike'
    full = model.full_velocity(model.vector(state))
    field, _ = model._boundary_quadrature.evaluate(full)
    un = np.sum(field*model._all_facets.normals, axis=0)
    inlet = np.isin(model._all_facets.find, mesh.boundaries['inlet'])[:, None]
    inward_power = .5*model.beta*float(np.sum(un*np.sum(field**2, axis=0)*model._all_facets.dx*inlet))
    trial = make_scheme(cfg['scheme'], cfg['gamma']).step(model, state, dt)
    after = float(np.max(np.hypot(trial.state.u[corner], trial.state.v[corner])))
    result: dict[str, Any] = dict(source_run=str(directory.resolve()),
        source_results_sha256=manifest['results_sha256'], inlet_treatment=inlet_treatment,
        backend=backend, dt=dt, before_time=state.t, after_time=trial.state.t,
        before_peak_speed=before, after_peak_speed=after, peak_growth_ratio=after/before,
        inlet_physical_convective_power=inward_power, r_before=state.r, r_after=trial.state.r,
        max_stage_residual=max(s.residual for s in trial.stages),
        max_stage_continuity=max(s.continuity_residual for s in trial.stages),
        passed=after <= before*(1+1e-8),
        qualification='One-step spike replay only; not a long-time or accuracy qualification.')
    assert digest(directory/'results.h5') == manifest['results_sha256']
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inlet-treatment', default='natural')
    parser.add_argument('--backend', choices=['direct', 'recycled'], default='direct')
    parser.add_argument('--dt', type=float, default=1e-7)
    parser.add_argument('--replay-to', type=float)
    args = parser.parse_args()
    result = probe(args.run, inlet_treatment=args.inlet_treatment, backend=args.backend,
                   dt=args.dt, replay_to=args.replay_to)
    label = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    output = PROJECT/'runs/hele_shaw_inlet_probe'/label
    output.mkdir(parents=True)
    write_json(output/'result.json', result)
    print(json.dumps(result), flush=True)
    print(output, flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

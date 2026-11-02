"""Compute explicit pressure-driven records; analysis is a separate entry point."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import h5py
import numpy as np

from experiments.fem_ns.attempts import save_attempts
from experiments.fem_ns.representation import maps
from experiments.periodic_ns.run import make_controller
from experiments.workflow import PROJECT, digest, provenance, save_result, write_json
from experiments.fem_backends import model_solver_options, solver_provenance
from solver.adaptivity import AdaptiveResult, ScalarErrorControl, integrate_adaptive
from solver.core import State, Trial
from solver.fem.airfoil import airfoil_polygon, make_airfoil_mesh, validate_airfoil_mesh
from solver.fem.darcy import solve_darcy
from solver.fem.hele_shaw import HeleShawModel
from solver.fem.space import MeshSpace
from solver.integrate import Result, integrate
from solver.schedule import schedule_nodes
from solver.schemes.registry import make_scheme, scheme_info
from .config import METADATA, effective_config
from .records import identity, load


class _AlignedFixed:
    """Give each fixed stage its exact declared start, including a short final step."""
    def __init__(self, scheme: Any, nodes: np.ndarray) -> None:
        self.scheme, self.nodes, self.index, self.name = scheme, nodes, 0, scheme.name

    def step(self, model: Any, state: State, dt: float) -> Trial:
        trial = self.scheme.step(model, state, dt)
        self.index += 1
        trial.state = State(float(self.nodes[self.index]), trial.state.u, trial.state.v, trial.state.r)
        return trial


def run(config: dict[str, Any], *, root: Path = PROJECT, rerun: bool = False) -> Path:
    cfg = effective_config(config)
    source = provenance()
    solver_provenance(source, cfg)
    source['packages'].update({k: version(k) for k in ('scikit-fem', 'gmsh', 'pymetis')})
    key = identity(cfg, source)
    parent = Path(root)/'runs/hele_shaw_airfoil'
    parent.mkdir(parents=True, exist_ok=True)
    if not rerun:
        for candidate in sorted(parent.iterdir()):
            if candidate.is_dir():
                try:
                    old, manifest, *_ = load(candidate)
                    if old == cfg and manifest['identity'] == key:
                        return candidate
                except (ValueError, KeyError, OSError):
                    continue
    directory = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+key[:8]+'-'+uuid4().hex[:8])
    directory.mkdir()
    write_json(directory/'config.json', cfg)
    manifest = dict(experiment='hele_shaw_airfoil', run_id=directory.name, identity=key,
                    source=source, status='running', config_sha256=digest(directory/'config.json'),
                    started_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory/'manifest.json', manifest)
    started = perf_counter()
    with (directory/'run.log').open('w', buffering=1) as log:
        log.write(json.dumps(cfg)+'\n')
        try:
            mesh = make_airfoil_mesh(chord=cfg['chord'], angle=cfg['angle'], gap_ratio=cfg['gap_ratio'],
                bounds=tuple(cfg['bounds']), surface_points=cfg['surface_points'],
                layer_cells=cfg['layer_cells'], far_size=cfg['far_size'],
                wake_region=None if 'wake_region' not in cfg else tuple(cfg['wake_region']),
                wake_size=cfg.get('wake_size'))
            mesh_check = validate_airfoil_mesh(mesh, airfoil_polygon(cfg['chord'], cfg['angle'], cfg['surface_points']),
                                                tuple(np.asarray(cfg['bounds'])*cfg['chord']))
            model = HeleShawModel(MeshSpace(mesh, cfg['quadrature_order']), cfg['nu'], cfg['gap'],
                rho=cfg['rho'], pressure_drop=cfg['pressure_drop'], ramp_time=cfg['ramp_time'],
                convection=cfg['convection'], outlet_treatment=cfg['outlet_treatment'],
                inlet_treatment=cfg.get('inlet_treatment', 'natural'),
                tolerance=cfg['stokes_tolerance'], cache_size=cfg.get('cache_size', 2), stokes_backend=cfg['stokes_backend'],
                stokes_ordering=cfg['stokes_ordering'], **model_solver_options(cfg))
            log.write('mesh '+json.dumps(mesh_check)+'\n')
            initial = model.state(0., model.zero_velocity(), cfg['r0'])
            scheme = make_scheme(cfg['scheme'], cfg['gamma'])
            observed = 0
            locations = model.velocity_basis.split_bases()[0].doflocs
            xmin, xmax, ymin, ymax = np.asarray(cfg['bounds'])*cfg['chord']
            corner = (locations[0] < xmin+.05*cfg['chord']) & (
                (locations[1] < ymin+.05*cfg['chord']) | (locations[1] > ymax-.05*cfg['chord']))

            def observe(state: State) -> None:
                nonlocal observed
                observed += 1
                speed = np.hypot(state.u, state.v)
                peak = float(np.max(speed))
                violation = (('velocity_limit' in cfg and peak > cfg['velocity_limit']) or
                             ('scalar_limit' in cfg and abs(state.r) > cfg['scalar_limit']))
                if observed == 1 or observed % cfg['log_every'] == 0 or state.t == cfg['T'] or violation:
                    write_json(directory/'progress.json', dict(status='guard_triggered' if violation else 'running',
                        accepted_steps=observed, t=state.t, target_time=cfg['T'], r=state.r,
                        maximum_nodal_speed=peak, inlet_corner_speed=float(np.max(speed[corner], initial=0.)),
                        workflow_wall_seconds=perf_counter()-started,
                        updated_utc=datetime.now(timezone.utc).isoformat()))
                if violation:
                    raise RuntimeError(f'Runtime qualification limit exceeded: speed={peak:.8g}, r={state.r:.8g}')

            observer = observe if any(k in cfg for k in ('inlet_treatment', 'velocity_limit', 'scalar_limit')) else None

            def progress(count: int, t: float, seconds: float) -> None:
                if count % cfg['log_every'] == 0 or t == cfg['T']:
                    log.write(f'accepted={count} t={t:.17g} integration_wall={seconds:.3f}s\n')

            result: Result
            if cfg['mode'] == 'adaptive':
                scalar = None if cfg['scalar_atol'] is None else ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
                result = integrate_adaptive(model, scheme, initial, cfg['T'], cfg['initial_step'],
                    controller=make_controller(cfg['controller'], cfg['scheme']), scalar_control=scalar,
                    snapshots=cfg['snapshots'], progress=progress, on_accept=observer)
            else:
                aligned = _AlignedFixed(scheme, schedule_nodes(0., np.asarray(cfg['actual_steps'])))
                result = integrate(model, aligned, initial, cfg['actual_steps'],
                                   snapshots=cfg['snapshots'], progress=progress, on_accept=observer)
            info = scheme_info(cfg['scheme'])
            metrics = {**model.performance_metrics(), **mesh_check,
                       'integration_wall_seconds': result.seconds,
                       'accepted_steps': len(result.times)-1,
                       'rejected_steps': sum(not a['accepted'] for a in result.attempts)
                           if isinstance(result, AdaptiveResult) else 0}
            pending = directory/'results.pending.h5'
            save_result(pending, result, metrics, n_stages=info.n_stages, n_roots=info.n_roots,
                        pressure_semantics=cfg['pressure_semantics'])
            with h5py.File(pending, 'a') as data:
                for name, value in METADATA.items():
                    value = cfg[name]
                    data.attrs[name] = json.dumps(value) if isinstance(value, list) else value
                if 'inlet_treatment' in cfg:
                    data.attrs['inlet_treatment'] = cfg['inlet_treatment']
                data.attrs['scikit_fem_version'] = version('scikit-fem')
                mapping = maps(model.velocity_basis, model.pressure_basis)
                mapping.update({'dofs/boundary': model.boundary_dofs, 'dofs/free': model.free_dofs})
                for name, values in mapping.items():
                    data.create_dataset(name, data=values)
                for name, facets in mesh.boundaries.items():
                    data.create_dataset('boundaries/'+name, data=facets)
                pressures, forces = [], []
                for state in result.snapshots:
                    velocity = model.vector(state)
                    _, pressure = model.instantaneous_solution(velocity, state.t)
                    pressures.append(model.physical_pressure(pressure))
                    forces.append(cfg['pressure_scale']*model.boundary_force(velocity, pressure, 'airfoil'))
                data.attrs['snapshot_pressure_semantics'] = 'instantaneous_physical_MWR_model_pressure'
                data.create_dataset('snapshots/physical_pressure', data=np.asarray(pressures).reshape(-1, model.pressure_basis.N))
                data.create_dataset('snapshots/model_force_per_span', data=np.asarray(forces).reshape(-1, 2))
                if cfg['darcy_reference']:
                    darcy = solve_darcy(mesh, cfg['nu'], cfg['gap'], cfg['pressure_drop'], rho=cfg['rho'],
                                        quadrature_order=cfg['quadrature_order'])
                    for name in ('pressure', 'u', 'v'):
                        data.create_dataset('darcy/'+name, data=getattr(darcy, name))
                    data['darcy'].attrs['fluxes'] = json.dumps(darcy.fluxes)
                    data['darcy'].attrs['residual'] = darcy.residual
                    data['darcy'].attrs['semantics'] = 'steady_target_pressure_scalar_P2_Darcy_reference'
                if isinstance(result, AdaptiveResult):
                    save_attempts(data, result.attempts)
            pending.replace(directory/'results.h5')
            manifest.update(status=result.status, error=result.error, final_time=result.final.t,
                results_sha256=digest(directory/'results.h5'), metrics=metrics,
                workflow_wall_seconds=perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(directory/'manifest.json', manifest)
            if observer is not None:
                live = json.loads((directory/'progress.json').read_text()) if (directory/'progress.json').exists() else {}
                live.update(status=result.status, error=result.error, t=result.final.t,
                            updated_utc=datetime.now(timezone.utc).isoformat())
                write_json(directory/'progress.json', live)
            log.write(json.dumps({'status': result.status, 'metrics': metrics, 'error': result.error})+'\n')
        except Exception as error:
            manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                            finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(directory/'manifest.json', manifest)
            log.write(manifest['error']+'\n')
            raise
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    directory = run(json.loads(args.config.read_text()), root=args.root, rerun=args.rerun)
    print(directory)
    _, manifest, *_ = load(directory, require_complete=False)
    if manifest['status'] != 'complete':
        raise SystemExit(manifest['error'])


if __name__ == '__main__':
    main()

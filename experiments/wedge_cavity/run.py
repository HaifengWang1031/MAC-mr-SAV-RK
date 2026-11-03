"""Run one lid-driven wedge cavity and record it.

The loop is written out instead of calling `solver.integrate` because each frame
of the animation needs the stage pressure at that time, and the stage pressure is
only available from the trial that produced the state.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from experiments.records import PROJECT, begin_run, digest, finish_run, provenance, run_identity
from experiments.fem_backends import model_solver_options, solver_provenance
from solver.core import State
from solver.fem.cavity import ClosedCavityModel
from solver.fem.wedge import WedgeSpace
from solver.schemes.registry import make_scheme, scheme_info
from .config import effective_config

EXPERIMENT = 'wedge_cavity'


def build(config: dict) -> tuple[dict, WedgeSpace, ClosedCavityModel]:
    cfg = effective_config(config)
    space = WedgeSpace(lid_length=cfg['lid_length'], apex_angle=cfg['apex_angle'],
                       lid_cells=cfg['lid_cells'], quadrature_order=cfg['quadrature_order'])
    model = ClosedCavityModel(space, cfg['nu'], lid_speed=cfg['lid_speed'],
                              stokes_backend=cfg['stokes_backend'],
                              tolerance=cfg['stokes_tolerance'], corner_policy=cfg['lid_corner_policy'],
                              **model_solver_options(cfg))
    return cfg, space, model


def steady_state_metrics(model: ClosedCavityModel, history: list[tuple[float, State]]) -> dict[str, float]:
    """Last-interval change of the state, normalised by the lid speed and the interval."""
    if len(history) < 2:
        return {}
    (t0, first), (t1, last) = history[-2], history[-1]
    if t1 <= t0:
        return {}
    difference = model.combine((1., model.vector(last)), (-1., model.vector(first)))
    scale = 1.+max(model.max_abs(last.u), model.max_abs(last.v))
    return {'steady_rate_inf': float(model.max_abs(difference)/(t1-t0)/scale),
            'steady_interval': float(t1-t0)}


def run(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    started = perf_counter()
    cfg, space, model = build(config)
    source = provenance()
    solver_provenance(source, cfg)
    identity = run_identity(cfg, source)
    parent = Path(root)/'runs'/EXPERIMENT
    parent.mkdir(parents=True, exist_ok=True)
    if not rerun:
        for candidate in sorted(parent.iterdir()):
            manifest_path = candidate/'manifest.json'
            if not manifest_path.is_file():
                continue
            try:
                old = json.loads(manifest_path.read_text())
                if old.get('identity') == identity and old.get('status') == 'complete' \
                        and digest(candidate/'results.h5') == old.get('results_sha256'):
                    return candidate
            except (OSError, ValueError, KeyError):
                continue
    directory, manifest = begin_run(parent, cfg, source,
        metadata={'schema_version': 1, 'experiment': EXPERIMENT, 'rerun': rerun}, started_key='created_utc')
    run_id = directory.name

    scheme = make_scheme(cfg['scheme'], cfg['gamma'])
    info = scheme_info(cfg['scheme'])
    stride = max(1, int(round(cfg['snapshot_every']/cfg['dt']))) if cfg['snapshot_every'] else 0
    state = model.state(0., model.zero_velocity(), cfg['r0'])
    times = [0.]
    diagnostics = [model.diagnostics(state)]
    snapshots: list[State] = [state]
    pressures: list[np.ndarray | None] = [None]
    recent: list[tuple[float, State]] = [(0., state)]
    stage_rows: list[list[float]] = []
    status, error = 'complete', ''
    with (directory/'run.log').open('w', buffering=1) as log:
        log.write(f'run_id={run_id}\nidentity={identity}\n')
        log.write(f'apex_angle={cfg["apex_angle"]} lid_cells={cfg["lid_cells"]} '
                  f'reynolds={cfg["reynolds"]:.6g} steps={cfg["steps"]} dt={cfg["dt"]}\n')
        try:
            for step in range(cfg['steps']):
                trial = scheme.step(model, state, cfg['dt'])
                target = cfg['T'] if step+1 == cfg['steps'] else (step+1)*cfg['dt']
                state = State(target, trial.state.u, trial.state.v, trial.state.r)
                times.append(state.t)
                diagnostics.append(model.diagnostics(state))
                stage = trial.stages[-1]
                stage_rows.append([stage.residual, stage.divergence_inf, stage.scalar_residual,
                                   float(getattr(stage, 'continuity_residual', float('nan')))])
                if (stride and (step+1) % stride == 0) or step+1 == cfg['steps']:
                    snapshots.append(state)
                    pressures.append(np.asarray(stage.pressure).copy())
                recent.append((state.t, state))
                if len(recent) > 2:
                    recent.pop(0)
                if (step+1) % cfg['log_every'] == 0 or step+1 == cfg['steps']:
                    log.write(f'accepted={step+1}/{cfg["steps"]} t={state.t:.6g} '
                              f'ke={diagnostics[-1]["kinetic"]:.10g} '
                              f'div={stage.divergence_inf:.3e} elapsed={perf_counter()-started:.2f}s\n')
        except Exception as failure:  # noqa: BLE001 - a failed record must still be written
            status = 'failed'
            error = f'{type(failure).__name__}: {failure}'
            log.write(error+'\n')
        elapsed = perf_counter()-started
        metrics = {'reynolds': cfg['reynolds'], 'accepted_steps': len(times)-1,
                   'wall_seconds': elapsed, 'steps_per_second': (len(times)-1)/elapsed if elapsed > 0 else 0.,
                   'snapshot_count': len(snapshots), 'snapshot_stride': stride,
                   'final_kinetic': diagnostics[-1]['kinetic'],
                   'final_divergence_inf': diagnostics[-1]['divergence_inf'],
                   'max_stage_residual': float(np.max([r[0] for r in stage_rows])) if stage_rows else 0.,
                   'max_continuity_residual': float(np.nanmax([r[3] for r in stage_rows])) if stage_rows else 0.,
                   **steady_state_metrics(model, recent)}
        metrics.update({key: value for key, value in model.performance_metrics().items()
                        if key in ('stokes_factorizations', 'stokes_fallbacks', 'stokes_krylov_iterations',
                                   'stokes_max_continuity_residual')})
        temporary = (directory/'results.h5').with_suffix('.tmp')
        with h5py.File(temporary, 'w') as data:
            data.attrs['schema_version'] = 1
            data.attrs['status'] = status
            data.attrs['metrics_json'] = json.dumps(metrics, allow_nan=False)
            data.attrs['experiment'] = EXPERIMENT
            data.attrs['scheme'] = cfg['scheme']
            data.attrs['n_stages'] = info.n_stages
            data.create_dataset('times', data=np.asarray(times))
            data.create_dataset('diagnostics/kinetic', data=[item['kinetic'] for item in diagnostics])
            data.create_dataset('diagnostics/divergence_inf', data=[item['divergence_inf'] for item in diagnostics])
            data.create_dataset('diagnostics/r', data=[item['r'] for item in diagnostics])
            data.create_dataset('stage_residual', data=np.asarray(stage_rows) if stage_rows
                                else np.zeros((0, 4)))
            data.create_dataset('snapshots/times', data=np.asarray([s.t for s in snapshots]))
            data.create_dataset('snapshots/u', data=np.stack([s.u for s in snapshots]), compression='gzip')
            data.create_dataset('snapshots/v', data=np.stack([s.v for s in snapshots]), compression='gzip')
            data.create_dataset('snapshots/r', data=np.asarray([s.r for s in snapshots]))
            pressure_values = [p for p in pressures if p is not None]
            if pressure_values:
                data.create_dataset('snapshots/pressure', data=np.stack(pressure_values), compression='gzip')
                data.create_dataset('snapshots/pressure_indices',
                                    data=np.asarray([index for index, p in enumerate(pressures) if p is not None]))
            data.create_dataset('mesh/points', data=space.mesh.p)
            data.create_dataset('mesh/triangles', data=space.mesh.t.T)
            data.create_dataset('mesh/lid', data=space.mesh.boundaries['lid'])
            data.create_dataset('mesh/wall_left', data=space.mesh.boundaries['wall_left'])
            data.create_dataset('mesh/wall_right', data=space.mesh.boundaries['wall_right'])
        temporary.replace(directory/'results.h5')
        finish_run(directory, manifest, status=status, error=error, metrics=metrics,
                   finished_key='finished_utc', fields={'accepted_steps': len(times)-1,
                                                       'final_time': float(times[-1])})
        log.write(json.dumps({'status': status, 'error': error, 'metrics': metrics})+'\n')
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a lid-driven wedge cavity')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = run(config, root=args.root, rerun=args.rerun)
    manifest = json.loads((directory/'manifest.json').read_text())
    print(directory)
    print(json.dumps({'status': manifest['status'], 'error': manifest.get('error', '')}, ensure_ascii=False))
    if manifest['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Compute periodic MHD benchmark runs with fixed steps and save their fields."""
import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import h5py
import numpy as np

from experiments.records import (
    PROJECT, begin_run, fail_run, finish_run, provenance, reusable_run, save_result, write_json,
)
from solver.core import State
from solver.fourier import FourierMHD, FourierSpace, MHDState
from solver.integrate import integrate, step_sizes
from solver.schedule import nearest_node_indices, schedule_nodes
from solver.schemes.registry import make_scheme, scheme_info
from .records import load_mhd
from .zhou import ZHOU_CASE, zhou_initial, magnetic_geometry

SCHEMES = ('sdirk3', 'sdirk3_mrsav')
DEFAULTS: dict[str, Any] = {
    'nx': 96, 'ny': 96, 'nu': .02, 'eta': .02, 'gamma': 1., 'T': 2.,
    'dt': 1/128, 'amplitude': 1., 'snapshots': [0., 1., 2.], 'scheme': 'sdirk3', 'log_every': 100,
}
ZHOU_DEFAULTS: dict[str, Any] = {
    'nx': 256, 'ny': 256, 'gamma': 1., 'T': 80., 'dt': 1/32,
    'snapshots': list(range(81)), 'scheme': 'sdirk3', 'log_every': 100,
    'S0': 125., 'noise_seed': 2019, 'noise_rms': 1e-4,
}


def orszag_tang_initial(model: FourierMHD, amplitude: float = 1.) -> MHDState:
    """Equation (7), with B = (A_y, -A_x), in Alfvén velocity units."""
    s = model.space
    if min(s.nx, s.ny) <= 6 or s.lx != 2*np.pi or s.ly != 2*np.pi:
        raise ValueError('Orszag–Tang requires a (2*pi)^2 box and a grid retaining mode 2')
    u = (-np.sin(s.y), np.sin(s.x))
    magnetic = (-np.sin(s.y), np.sin(2*s.x))
    return model.state(0., s.project(s.to_spectral(amplitude*np.asarray((u, magnetic)))))


def effective_config(config: dict) -> dict:
    case = config.get('case', 'orszag_tang_1979')
    if case not in ('orszag_tang_1979', ZHOU_CASE):
        raise ValueError(f'Unknown periodic MHD case: {case}')
    defaults = ZHOU_DEFAULTS if case == ZHOU_CASE else DEFAULTS
    if unknown := set(config)-set(defaults)-{'case'}:
        raise ValueError(f'Unknown periodic MHD configuration keys: {sorted(unknown)}')
    cfg = {**defaults, **config}
    if case == ZHOU_CASE:
        if not np.isfinite([cfg['S0'], cfg['noise_rms']]).all() or cfg['S0'] <= 0 \
                or cfg['noise_rms'] < 0 or type(cfg['noise_seed']) is not int or cfg['noise_seed'] < 0:
            raise ValueError('Zhou case requires positive S0, nonnegative noise RMS and integer seed')
        cfg['nu'] = cfg['eta'] = 1/(8*cfg['S0'])
    space = FourierSpace(cfg['nx'], cfg['ny'])
    if min(space.nx, space.ny) <= (24 if case == ZHOU_CASE else 6) or cfg['scheme'] not in SCHEMES:
        raise ValueError('Grid must retain the initial modes and use an SDIRK3 scheme')
    amplitude = cfg.get('amplitude', 1.)
    if not np.isfinite([cfg['nu'], cfg['eta'], cfg['gamma'], amplitude]).all() \
            or min(cfg['nu'], cfg['eta'], amplitude) <= 0 or cfg['gamma'] < 0:
        raise ValueError('Invalid MHD diffusivities, amplitude or damping')
    if type(cfg['log_every']) is not int or cfg['log_every'] < 1:
        raise ValueError('log_every must be a positive integer')
    requests = np.asarray(cfg['snapshots'], dtype=float)
    if requests.ndim != 1 or not len(requests) or not np.isfinite(requests).all():
        raise ValueError('Provide finite snapshot times')
    steps = step_sizes(cfg['T'], dt=cfg['dt'])
    nodes = schedule_nodes(0., steps)
    if np.any(np.diff(requests) <= 0) or requests[0] < 0 or requests[-1] > cfg['T'] \
            or not np.allclose(nodes[nearest_node_indices(nodes, requests.tolist())], requests, rtol=0, atol=1e-12):
        raise ValueError('MHD snapshot times must be increasing and align with fixed steps')
    cfg.update(space.metadata(), snapshots=requests.tolist(), case=case,
               pressure_semantics='leray_eliminated', r0=0.)
    return cfg


def _magnetic(state: State) -> np.ndarray:
    if not isinstance(state, MHDState):
        raise TypeError('Expected a saved MHD state')
    return state.magnetic


def run(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    cfg = effective_config(config)
    source = provenance()
    parent = root/'runs'/'periodic_mhd'
    reused = reusable_run(parent, cfg, source, reader=load_mhd, rerun=rerun)
    if reused is not None:
        return reused
    directory, manifest = begin_run(parent, cfg, source, metadata={'experiment': 'periodic_mhd'})
    log = directory/'run.log'
    log.write_text(f'Started {manifest["started"]}\n{json.dumps(cfg, indent=2)}\n')

    def progress(count: int, t: float, elapsed: float) -> None:
        if count % cfg['log_every'] == 0 or abs(t-cfg['T']) < 1e-12:
            with log.open('a') as stream:
                stream.write(f'accepted={count} t={t:.17g} wall={elapsed:.3f}s\n')

    try:
        model = FourierMHD(FourierSpace(cfg['nx'], cfg['ny']), cfg['nu'], cfg['eta'])
        initial = (zhou_initial(model, cfg['noise_seed'], cfg['noise_rms']) if cfg['case'] == ZHOU_CASE
                   else orszag_tang_initial(model, cfg['amplitude']))
        result = integrate(model, make_scheme(cfg['scheme'], cfg['gamma']),
                           initial, step_sizes(cfg['T'], dt=cfg['dt']),
                           snapshots=cfg['snapshots'], progress=progress)
        metrics = {'integration_wall_seconds': result.seconds, 'accepted_steps': len(result.times)-1}
        pending = directory/'results.pending.h5'
        info = scheme_info(cfg['scheme'])
        save_result(pending, result, metrics, n_stages=info.n_stages, n_roots=info.n_roots,
                    pressure_semantics=cfg['pressure_semantics'], include_stage_pressure=False)
        with h5py.File(pending, 'a') as data:
            data.create_dataset('final/magnetic', data=_magnetic(result.final))
            magnetic = data.create_dataset('snapshots/magnetic',
                                           shape=(len(result.snapshots), 2, *model.space.shape),
                                           dtype='f8', compression='gzip')
            if cfg['case'] == ZHOU_CASE:
                flux = data.create_dataset('snapshots/flux', shape=(len(magnetic), *model.space.shape),
                                           dtype='f8', compression='gzip')
                current = data.create_dataset('snapshots/current', shape=flux.shape,
                                              dtype='f8', compression='gzip')
                for index, snapshot in enumerate(result.snapshots):
                    b = _magnetic(snapshot)
                    magnetic[index] = b
                    flux[index], current[index] = magnetic_geometry(model.space, b)
            else:
                for index, snapshot in enumerate(result.snapshots):
                    magnetic[index] = _magnetic(snapshot)
        pending.replace(directory/'results.h5')
        finish_run(directory, manifest, status=result.status, error=result.error, metrics=metrics)
    except Exception as error:
        fail_run(directory, manifest, error)
        with log.open('a') as stream:
            stream.write(manifest['error']+'\n')
        raise
    with log.open('a') as stream:
        stream.write(f'{result.status}: {result.error}\n{json.dumps(metrics, indent=2)}\n')
    return directory


def run_study(config: dict, *, root: Path = PROJECT) -> Path:
    """Run fixed-grid time refinement, a reference sensitivity check and paper curves."""
    directory = root/'runs'/'periodic_mhd_studies'/uuid4().hex[:12]
    directory.mkdir(parents=True)
    write_json(directory/'config.json', config)
    base = config['physical']
    grid, curve_grid = config['time_grid'], config['curve_grid']
    curve_dt = config['curve_dt']
    amplitude = base.get('amplitude', DEFAULTS['amplitude'])
    curve_amplitudes = config.get('curve_amplitudes', [amplitude])
    members = [
        ('reference', 'sdirk3', grid, config['reference_dt'], amplitude),
        ('reference_check', 'sdirk3', grid, 2*config['reference_dt'], amplitude),
        *[('convergence', name, grid, dt, amplitude) for name in SCHEMES for dt in config['time_steps']],
    ]
    for a in curve_amplitudes:
        members.extend([
            *[('curves', name, curve_grid, curve_dt, a) for name in SCHEMES],
            ('curve_grid_check', 'sdirk3', grid, curve_dt, a),
            ('curve_step_check', 'sdirk3', curve_grid, 2*curve_dt, a),
        ])
    batch: dict[str, Any] = {'status': 'running', 'members': []}
    batch_file = directory/'batch.json'
    write_json(batch_file, batch)
    print(batch_file, flush=True)
    try:
        for role, name, n, dt, a in members:
            path = run({**base, 'nx': n, 'ny': n, 'dt': dt, 'scheme': name, 'amplitude': a}, root=root)
            batch['members'].append({'role': role, 'run': str(path.resolve())})
            write_json(batch_file, batch)
            _, manifest = load_mhd(path)
            if manifest['status'] != 'complete':
                raise RuntimeError(f'{path.name}: {manifest["error"]}')
            print(f'{role} {name} {n}² dt={dt:.9g} amplitude={a:g} {path.name} '
                  f'{manifest["metrics"]["integration_wall_seconds"]:.2f}s', flush=True)
    except Exception as error:
        batch.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(batch_file, batch)
        raise
    batch['status'] = 'complete'
    write_json(batch_file, batch)
    return batch_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--study', action='store_true')
    parser.add_argument('--scheme', choices=SCHEMES)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.study:
        if args.scheme is not None:
            parser.error('--scheme applies to single runs')
        run_study(config, root=args.root)
    else:
        if args.scheme is not None:
            config['scheme'] = args.scheme
        path = run(config, root=args.root)
        print(path)
        load_mhd(path, require_complete=True)


if __name__ == '__main__':
    main()

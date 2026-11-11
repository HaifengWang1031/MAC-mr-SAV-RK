"""Bounded smoke tests and native concurrency benchmarks, never a production launcher.

128 denotes the nonlinear/observable Gauss rule, not 128 velocity modes.
The maximum compatible dealiased Dirichlet modal count is 84.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import monotonic, perf_counter, sleep
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info

from experiments.controllers import controller_settings, make_controller
from experiments.records import digest, provenance, write_json
from solver.adaptivity import ScalarErrorControl, integrate_adaptive
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV
from solver.spectral.assembly import Space
from solver.spectral.model import SpectralModel

PHASE_DEGREES = (0., 15., 22.5, 30., 45., 60., 75., 90., 105., 120., 135., 150., 180.)
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')
PRODUCTION = {'T': 100000., 'scalar_interval': 1., 'field_interval': 10.,
              'gauss_nodes': 128, 'requires_explicit_approval': True}


def modes_for_gauss(nodes: int) -> int:
    """Largest modal count satisfying 2q-1 >= 3(N+1), no underintegration."""
    if nodes < 8:
        raise ValueError('At least eight Gauss nodes required')
    return (2*nodes-1)//3-1


def make_model(nodes: int, case: int, backend: str = 'tensor') -> SpectralModel:
    """Build the paper's stationary force, projected with the specified Gauss rule."""
    if case not in range(13):
        raise ValueError('Case must be X0..X12')
    size = modes_for_gauss(nodes)
    model = SpectralModel(Space(size, 2*np.pi, 2*np.pi), 2e-4,
                          dealias=nodes-size, stokes_backend=backend,
                          tolerance=1e-10, krylov_tolerance=1e-12)
    phi = np.deg2rad(PHASE_DEGREES[case])
    force = np.r_[model.project(lambda x, y: .1*np.sin(5*y), extra=nodes-size),
                  model.project(lambda x, y: .1*np.sin(5*x+phi), extra=nodes-size)]
    model.force = lambda t: force
    return model


def observables(model: SpectralModel, state: Any, case: int) -> dict[str, float]:
    """Area averages from physical quadrature, including unprojected applied torque."""
    space = model.space
    x, wx, y, wy = space.nodes(model.quadrature_extra)
    xx, yy = np.meshgrid(x, y)
    weight = wy[:, None]*wx[None, :]/(space.lx*space.ly)
    velocity = model.vector(state).reshape(2, space.size, space.size)
    u, v = (space.evaluate(c, x, y) for c in velocity)
    bx, by = space.velocity_values(x, y)
    dx, dy = space.velocity_derivatives(x, y)
    ux, vx = (by.T @ c.T @ dx for c in velocity)
    uy, vy = (dy.T @ c.T @ bx for c in velocity)
    gradient_squared = float(np.sum(weight*(ux*ux+uy*uy+vx*vx+vy*vy)))
    divergence_squared = float(np.sum(weight*(ux+vy)**2))
    phi = np.deg2rad(PHASE_DEGREES[case])
    fx, fy = .1*np.sin(5*yy), .1*np.sin(5*xx+phi)
    return {'enstrophy': float(np.sum(weight*(vx-uy)**2)/2),
            'dissipation': model.nu*gradient_squared,
            'divergence_rms': float(np.sqrt(divergence_squared)),
            'relative_divergence': float(np.sqrt(divergence_squared/gradient_squared))
                                   if gradient_squared else 0.,
            'energy': float(np.sum(weight*(u*u+v*v))/2),
            'angular_momentum': float(np.sum(weight*((xx-np.pi)*v-(yy-np.pi)*u))),
            'power': float(np.sum(weight*(fx*u+fy*v))),
            'torque': float(np.sum(weight*((xx-np.pi)*fy-(yy-np.pi)*fx)))}


def run_probe(output: Path, *, nodes: int = 128, case: int = 4,
              duration: float = 1., backend: str = 'tensor', seed: Path | None = None,
              barrier: Path | None = None) -> dict[str, Any]:
    """Save one short accepted-prefix probe. No claim of production checkpointing."""
    if not 0 < duration <= 100:
        raise ValueError('Preflight duration must be in (0,100]; production is not supported')
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    model = make_model(nodes, case, backend)
    if seed is None:
        initial = model.state(0., model.zero_velocity(), 0.)
    else:
        with np.load(seed) as data:
            if int(data['case']) != case or int(data['nodes']) != nodes:
                raise ValueError('Seed physical/spatial configuration mismatch')
            initial = model.state(float(data['t']), data['velocity'], float(data['r']))
    # Warm caches outside timed integration; never commit this trial.
    SDIRK3MRSAV(1000.).step(model, initial, .001)
    setup = perf_counter()-start
    if barrier is not None:
        (output/'ready').touch()
        deadline = monotonic()+180
        while not barrier.exists():
            if monotonic() > deadline:
                raise TimeoutError('Benchmark release barrier timed out')
            sleep(.01)
    controller = make_controller({'kind': 'PI', 'atol': 1e-8, 'rtol': 1e-5,
                                  'min_step': 1e-12, 'max_step': .05,
                                  'min_factor': .5, 'max_factor': 1.2,
                                  'max_steps': 200000}, 'sdirk3_mrsav')
    accepted_count = 0

    def progress_state(state: Any) -> None:
        nonlocal accepted_count
        accepted_count += 1
        if accepted_count % 100 == 0:
            write_json(output/'progress.json', {'t': state.t, 'accepted_steps': accepted_count,
                                               'wall_seconds': perf_counter()-start})
            np.savez(output/'latest.npz', velocity=model.vector(state), r=state.r,
                     t=state.t, nodes=nodes, case=case)

    requests = [float(k) for k in range(int(np.ceil(initial.t)),
                                        int(np.floor(initial.t+duration))+1)]
    result = integrate_adaptive(model, SDIRK3MRSAV(1000.), initial, duration, .001,
                                controller=controller, scalar_control=ScalarErrorControl(1e-5, 1000.),
                                snapshots=requests, strict_snapshots=False, on_accept=progress_state)
    vector = model.vector(result.final)
    np.savez(output/'final.npz', velocity=vector, r=result.final.r, t=result.final.t,
             nodes=nodes, case=case)
    np.savez(output/'history.npz', times=result.times,
             steps=[a['step'] for a in result.attempts],
             accepted=[a['accepted'] for a in result.attempts])
    samples = [{'request_t': request, 'actual_t': actual,
                **observables(model, state, case)}
               for request, actual, state in zip(result.snapshot_requests,
                                                 result.snapshot_times, result.snapshots)]
    write_json(output/'scalar_samples.json', {'samples': samples})
    field_indices = [i for i, request in enumerate(result.snapshot_requests) if request % 10 == 0]
    np.savez(output/'field_samples.npz',
             requests=np.array(result.snapshot_requests)[field_indices],
             times=np.array(result.snapshot_times)[field_indices],
             velocity=np.array([model.vector(result.snapshots[i]) for i in field_indices]))
    stages = [s for row in result.stages for s in row]
    row = {'status': result.status, 'error': result.error, 'gauss_nodes': nodes,
           'source': provenance(),
           'controller': controller_settings(controller), 'initial_step': .001,
           'gamma': 1000., 'scalar_atol': 1e-5,
           'seed': None if seed is None else {'path': str(seed), 'sha256': digest(seed)},
           'artifacts': {p.name: digest(p) for p in output.iterdir() if p.suffix in ('.npz', '.json')
                         and p.name != 'progress.json'}, 
           'modes': model.space.size, 'case': case, 'backend': backend,
           'phase_degrees': PHASE_DEGREES[case], 'phase_source': 'inferred_from_rounded_table',
           'production_target': PRODUCTION, 'start_t': initial.t, 'final_t': result.final.t,
           'setup_wall_seconds': setup, 'integration_wall_seconds': result.seconds,
           'accepted_steps': len(result.times)-1, 'attempts': len(result.attempts),
           'max_coupling_deviation': max((abs(s['frozen_factor']*(1-s['r'])-1)
                                          for s in stages), default=0.),
           'max_stage_residual': max((s['residual'] for s in stages), default=0.),
           'max_continuity_residual': max((s['continuity_residual'] for s in stages), default=0.),
           'max_physical_divergence': max(d['divergence_inf'] for d in result.diagnostics),
           'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*
                             (1 if sys.platform == 'darwin' else 1024),
           'threadpools': threadpool_info(),
           'thread_environment': {k: os.environ.get(k) for k in THREAD_KEYS},
           'final_observables': observables(model, result.final, case),
           'qualification': 'short_probe_only; PI history reset at seed; not production resume'}
    write_json(output/'result.json', row)
    model.close()
    if result.status != 'complete':
        raise RuntimeError(f'Probe failed: {result.error}')
    return row


def benchmark(output: Path, seed: Path, *, duration: float = .2, repeats: int = 3,
              combinations: list[tuple[int, int]] | None = None) -> None:
    """Interleave native worker/thread combinations with a synchronized integration start."""
    output.mkdir(parents=True, exist_ok=False)
    combinations = combinations or [(1, 1), (2, 1), (4, 1), (6, 1), (8, 1), (2, 2), (4, 2)]
    if any(w < 1 or w > 13 or t < 1 or t > 4 for w, t in combinations):
        raise ValueError('Benchmark requires 1..13 workers and 1..4 threads')
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        order = combinations if repeat % 2 == 0 else list(reversed(combinations))
        for workers, threads in order:
            group = output/f'r{repeat}-w{workers}-t{threads}'
            group.mkdir()
            release = group/'go'
            processes = []
            streams = []
            try:
                for worker in range(workers):
                    env = dict(os.environ, **{key: str(threads) for key in THREAD_KEYS})
                    stream = (group/f'{worker}.log').open('w')
                    streams.append(stream)
                    command = [sys.executable, '-m', 'experiments.torque_transition.preflight',
                               'probe', '--output', str(group/str(worker)), '--seed', str(seed),
                               '--duration', str(duration), '--barrier', str(release)]
                    processes.append(subprocess.Popen(command, env=env, stdout=stream, stderr=stream))
                deadline = monotonic()+180
                while not all((group/str(i)/'ready').exists() for i in range(workers)):
                    if any(p.poll() is not None for p in processes) or monotonic() > deadline:
                        raise RuntimeError(f'Worker setup failed: {group}')
                    sleep(.02)
                start = perf_counter()
                release.touch()
                for p in processes:
                    if p.wait(timeout=180) != 0:
                        raise RuntimeError(f'Worker integration failed: {group}')
                group_wall = perf_counter()-start
                results = [json.loads((group/str(i)/'result.json').read_text()) for i in range(workers)]
                # Same seed/algorithm must produce equal trajectories irrespective of concurrency.
                with np.load(group/'0/final.npz') as reference:
                    for i in range(1, workers):
                        with np.load(group/str(i)/'final.npz') as actual:
                            np.testing.assert_allclose(actual['velocity'], reference['velocity'], rtol=1e-9, atol=1e-11)
                row = {'repeat': repeat, 'workers': workers, 'threads': threads,
                       'integration_seconds_max': max(r['integration_wall_seconds'] for r in results),
                       'release_to_exit_seconds': group_wall,
                       'throughput': workers*duration/group_wall,
                       'rss_sum_peak_bytes': sum(r['peak_rss_bytes'] for r in results),
                       'max_stage_residual': max(r['max_stage_residual'] for r in results)}
                rows.append(row)
                write_json(output/'measurements.json', {'production_target': PRODUCTION, 'rows': rows})
                print(json.dumps(row), flush=True)
            finally:
                for p in processes:
                    if p.poll() is None:
                        p.terminate()
                        p.wait()
                for stream in streams:
                    stream.close()
    summary = []
    for workers, threads in combinations:
        selected = [r for r in rows if r['workers'] == workers and r['threads'] == threads]
        summary.append({'workers': workers, 'threads': threads,
                        'median_throughput': float(np.median([r['throughput'] for r in selected])),
                        'max_peak_rss_bytes': max(r['rss_sum_peak_bytes'] for r in selected)})
    write_json(output/'summary.json', {'status': 'complete', 'rows': sorted(summary, key=lambda r: -r['median_throughput']),
                                     'production_target': PRODUCTION})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe', 'benchmark'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=1.)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--barrier', type=Path)
    parser.add_argument('--case', type=int, default=4)
    parser.add_argument('--nodes', type=int, default=128)
    parser.add_argument('--backend', choices=('tensor', 'direct'), default='tensor')
    parser.add_argument('--combinations', help='Additional bounded candidates, e.g. 10x1,12x1,13x1')
    args = parser.parse_args()
    if args.mode == 'benchmark':
        if args.seed is None:
            parser.error('benchmark requires --seed')
        combinations = None if args.combinations is None else [
            (int(pair.split('x')[0]), int(pair.split('x')[1]))
            for pair in args.combinations.split(',')]
        benchmark(args.output, args.seed, duration=args.duration, combinations=combinations)
    else:
        print(json.dumps(run_probe(args.output, nodes=args.nodes, case=args.case,
                                   duration=args.duration, backend=args.backend,
                                   seed=args.seed, barrier=args.barrier)))


if __name__ == '__main__':
    main()

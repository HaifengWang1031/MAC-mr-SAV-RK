"""Explicitly approved MAC256 production: bounded histories, natural-node streaming, logs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from threading import Event, RLock
from time import perf_counter
from typing import Any, TextIO

import numpy as np

from experiments.controllers import make_controller
from experiments.records import digest, provenance
from experiments.torque_transition.mac_model import make_mac_model, mac_observables
from experiments.torque_transition.preflight import PHASE_DEGREES, THREAD_KEYS
from solver.adaptivity import AdaptiveContinuation, AdaptiveResult, ScalarErrorControl, integrate_adaptive
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV

FINAL_T = 10000.


class StopFlag:
    """Idempotent signal flag: no lock acquisition inside a signal handler."""

    def __init__(self) -> None:
        self.requested = False

    def set(self) -> None:
        self.requested = True

    def is_set(self) -> bool:
        return self.requested


ATTEMPT_KEYS = ('t', 'step', 'error', 'accepted', 'error_u', 'error_r', 'scalar_defect',
                'reason', 'factor', 'boundary_partition', 'output_truncated',
                'previous_accepted_error', 'previous_accepted_error_u', 'previous_accepted_error_r',
                'controller', 'elapsed_seconds', 'cpu_seconds')
SCALAR_KEYS = ('request_t', 'actual_t', 'r', 'energy', 'angular_momentum', 'power', 'dissipation',
               'enstrophy', 'dissipation_minus_2nuZ', 'divergence_rms', 'relative_divergence', 'torque')
STAGE_KEYS = ('step_index', 't', 'r', 'max_G_deviation', 'stage_residual',
              'continuity_residual', 'divergence_inf')


def event(kind: str, **values: Any) -> None:
    print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'event': kind,
                      **values}, ensure_ascii=False), flush=True)


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def atomic_npz(path: Path, **data: Any) -> None:
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


class Streams:
    """Append-only journals; a checkpoint binds exact durable byte offsets."""

    def __init__(self, directory: Path, offsets: dict[str, int] | None) -> None:
        self.files: dict[str, TextIO] = {}
        self.writers: dict[str, Any] = {}
        for name, keys in (('attempts.csv', ATTEMPT_KEYS), ('scalars.csv', SCALAR_KEYS),
                           ('stages.csv', STAGE_KEYS), ('fields.jsonl', ())):
            path = directory/name
            if offsets is None:
                stream = path.open('x', newline='', encoding='utf-8')
            else:
                if not path.exists() or path.stat().st_size < offsets[name]:
                    raise ValueError(f'Missing/truncated checkpoint journal: {path}')
                stream = path.open('r+', newline='', encoding='utf-8')
                stream.seek(offsets[name])
                stream.truncate()
            self.files[name] = stream
            if keys:
                writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='raise')
                self.writers[name] = writer
                if offsets is None:
                    writer.writeheader()

    def durable_offsets(self) -> dict[str, int]:
        offsets = {}
        for name, stream in self.files.items():
            stream.flush()
            os.fsync(stream.fileno())
            offsets[name] = stream.tell()
        return offsets

    def close(self) -> None:
        for stream in self.files.values():
            stream.close()


class Recorder:
    """At most two accepted face states; no output-related timestep truncation."""

    def __init__(self, directory: Path, model: Any, case: int, plan_hash: str,
                 initial: Any, metadata: dict[str, Any] | None, stopping: Event | StopFlag) -> None:
        self.directory, self.model, self.case = directory, model, case
        self.plan_hash, self.previous, self.stopping = plan_hash, initial, stopping
        self.next_scalar = metadata['next_scalar'] if metadata else 0
        self.next_field = metadata['next_field'] if metadata else 0
        self.streams = Streams(directory, metadata['offsets'] if metadata else None)
        self.started, self.last_checkpoint = perf_counter(), perf_counter()
        self.wall_base = metadata['wall_seconds'] if metadata else 0.
        self.last_checkpoint_t = initial.t
        self.last_history = AdaptiveContinuation(**metadata['history']) if metadata else AdaptiveContinuation(1e-4)
        self.last_state = initial
        if metadata is None:
            self.sample(initial)
            self.checkpoint(initial, self.last_history)

    def sample(self, current: Any) -> None:
        previous = self.previous
        cache: dict[float, dict[str, float]] = {}
        while self.next_scalar <= FINAL_T and self.next_scalar <= current.t:
            request = float(self.next_scalar)
            selected = previous if request-previous.t <= current.t-request else current
            if selected.t not in cache:
                cache[selected.t] = mac_observables(self.model, selected, self.case)
            self.streams.writers['scalars.csv'].writerow(
                {'request_t': request, 'actual_t': selected.t, 'r': selected.r, **cache[selected.t]})
            self.next_scalar += 1
        while 10*self.next_field <= FINAL_T and 10*self.next_field <= current.t:
            request = float(10*self.next_field)
            selected = previous if request-previous.t <= current.t-request else current
            path = self.directory/'fields'/f'{self.next_field:05d}.npz'
            atomic_npz(path, velocity=self.model.vector(selected), r=selected.r, t=selected.t,
                       request_t=request, nodes=256, case=self.case, discretization='mac',
                       phase_degrees=PHASE_DEGREES[self.case], plan_sha256=self.plan_hash)
            self.streams.files['fields.jsonl'].write(json.dumps(
                {'request_t': request, 'actual_t': selected.t, 'path': str(path.relative_to(self.directory)),
                 'sha256': digest(path)})+'\n')
            self.next_field += 1
        self.previous = current

    def checkpoint(self, state: Any, history: AdaptiveContinuation) -> None:
        vector = self.model.vector(state)
        metadata = {'schema': 1, 'plan_sha256': self.plan_hash, 'case': self.case,
                    'nodes': 256, 't': state.t, 'r': state.r,
                    'velocity_sha256': hashlib.sha256(vector.tobytes()).hexdigest(),
                    'history': asdict(history), 'next_scalar': self.next_scalar,
                    'next_field': self.next_field, 'offsets': self.streams.durable_offsets(),
                    'wall_seconds': self.wall_base+perf_counter()-self.started}
        atomic_npz(self.directory/'checkpoint.npz', velocity=vector,
                   metadata=json.dumps(metadata, allow_nan=False))
        self.last_checkpoint, self.last_checkpoint_t = perf_counter(), state.t
        atomic_json(self.directory/'status.json', {'status': 'running', 'pid': os.getpid(),
                    't': state.t, 'accepted_steps': history.accepted_steps,
                    'attempted_steps': history.attempted_steps, 'checkpoint_t': state.t})
        event('checkpoint', case=self.case, t=state.t, steps=history.accepted_steps,
              attempts=history.attempted_steps)

    def advance(self, result: AdaptiveResult) -> None:
        assert result.continuation is not None
        state = result.final
        self.sample(state)
        stages = result.stages[-1]
        self.streams.writers['stages.csv'].writerow({
            'step_index': result.continuation.accepted_steps, 't': state.t, 'r': state.r,
            'max_G_deviation': max(abs(s['frozen_factor']*(1-s['r'])-1) for s in stages),
            'stage_residual': max(s['residual'] for s in stages),
            'continuity_residual': max(s['continuity_residual'] for s in stages),
            'divergence_inf': result.diagnostics[-1]['divergence_inf']})
        self.last_state, self.last_history = state, result.continuation
        if state.t-self.last_checkpoint_t >= 10 or perf_counter()-self.last_checkpoint >= 900 \
                or self.stopping.is_set() or state.t == FINAL_T:
            self.checkpoint(state, result.continuation)
            if shutil.disk_usage(self.directory).free < 20*1024**3:
                raise RuntimeError('Less than 20GiB disk free; paused at durable checkpoint')
        if self.stopping.is_set():
            raise RuntimeError('Stop requested; durable checkpoint saved')


def run_case(directory: Path, case: int, *, resume: bool = False) -> None:
    """One full trajectory; no pilot, no shortened production T and no forced acceptance."""
    plan_path = directory/'plan.json'
    plan, plan_hash = json.loads(plan_path.read_text()), digest(plan_path)
    if not plan['approved'] or plan['T'] != FINAL_T or plan['grid_cells'] != 256:
        raise ValueError('Missing production approval or changed production target')
    if provenance()['code_sha256'] != plan['source']['code_sha256']:
        raise ValueError('Frozen production source changed')
    if case not in range(13):
        raise ValueError('Expected case X0..X12')
    output = directory/f'X{case:02d}'
    output.mkdir(exist_ok=resume)
    (output/'fields').mkdir(exist_ok=resume)
    stopping = StopFlag()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.set())
    model = make_mac_model(256, PHASE_DEGREES[case])
    metadata = None
    if resume:
        with np.load(output/'checkpoint.npz', allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            vector = saved['velocity'].copy()
        allowed_plans = {plan_hash}
        if 'parent_plan_sha256' in plan:
            if digest(directory/'parent-plan.json') != plan['parent_plan_sha256']:
                raise ValueError('Retarget parent plan checksum mismatch')
            allowed_plans.add(plan['parent_plan_sha256'])
        if metadata['plan_sha256'] not in allowed_plans or metadata['case'] != case or metadata['nodes'] != 256:
            raise ValueError('Checkpoint identity mismatch')
        if not 0 <= metadata['t'] <= FINAL_T:
            raise ValueError('Checkpoint outside shortened production interval')
        if hashlib.sha256(vector.tobytes()).hexdigest() != metadata['velocity_sha256']:
            raise ValueError('Checkpoint velocity checksum mismatch')
        initial = model.state(metadata['t'], vector, metadata['r'])
    else:
        initial = model.state(0., np.zeros(model.grid.size), 0.)
    recorder = Recorder(output, model, case, plan_hash, initial, metadata, stopping)
    event('started', case=case, pid=os.getpid(), t=initial.t, final_T=FINAL_T, resume=resume)
    try:
        if initial.t < FINAL_T:
            controller = make_controller({**plan['controller'], 'max_steps': 100_000_000}, 'sdirk3_mrsav')
            result = integrate_adaptive(model, SDIRK3MRSAV(1000.), initial, FINAL_T-initial.t, 1e-4,
                controller=controller, scalar_control=ScalarErrorControl(1e-3, 1000.),
                min_step_policy='fail', retain_history=False,
                continuation=recorder.last_history if resume else None,
                on_attempt=recorder.streams.writers['attempts.csv'].writerow,
                on_advance=recorder.advance)
            if result.status != 'complete':
                raise RuntimeError(result.error)
        recorder.checkpoint(recorder.last_state, recorder.last_history)
        atomic_json(output/'status.json', {'status': 'complete', 'pid': os.getpid(), 't': FINAL_T,
                    'accepted_steps': recorder.last_history.accepted_steps,
                    'attempted_steps': recorder.last_history.attempted_steps,
                    'plan_sha256': plan_hash, 'checkpoint_sha256': digest(output/'checkpoint.npz')})
        event('completed', case=case, t=FINAL_T)
    except BaseException as exc:
        # Keep the last transactional checkpoint. Never relabel a failure as a completed trajectory.
        atomic_json(output/'status.json', {'status': 'paused' if stopping.is_set() else 'failed',
                    'pid': os.getpid(), 'checkpoint_t': recorder.last_checkpoint_t,
                    'error': f'{type(exc).__name__}: {exc}', 'plan_sha256': plan_hash})
        event('stopped', case=case, checkpoint_t=recorder.last_checkpoint_t, error=str(exc))
        raise
    finally:
        recorder.streams.close()


def retarget(parent: Path, directory: Path) -> None:
    """Copy a stopped prefix and explicitly amend only T; never relabel old payloads."""
    parent = parent.resolve()
    old_plan = json.loads((parent/'plan.json').read_text())
    old_hash = digest(parent/'plan.json')
    supervisor = parent/'supervisor.json'
    if supervisor.exists():
        pid = json.loads(supervisor.read_text()).get('pid')
        if pid and Path(f'/proc/{pid}').exists():
            raise ValueError(f'Parent supervisor still exists: {pid}')
    if not old_plan['approved'] or old_plan['T'] <= FINAL_T:
        raise ValueError('Expected an approved longer parent trajectory')
    old_root, new_root = parent.parents[1], Path(__file__).resolve().parents[2]
    paths = [Path(name) for name in ('pyproject.toml', 'uv.lock', '.python-version')]
    for name in ('solver', 'experiments', 'tools'):
        old_files = {p.relative_to(old_root) for p in (old_root/name).rglob('*.py')}
        new_files = {p.relative_to(new_root) for p in (new_root/name).rglob('*.py')}
        if old_files != new_files:
            raise ValueError('Source inventory changed outside the target amendment')
        paths.extend(sorted(old_files))
    fingerprint = hashlib.sha256()
    for path in paths:
        before = (old_root/path).read_bytes()
        fingerprint.update(str(path).encode())
        fingerprint.update(before)
        if str(path) != 'experiments/torque_transition/production.py' and before != (new_root/path).read_bytes():
            raise ValueError(f'Numerical/dependency source changed: {path}')
    if fingerprint.hexdigest() != old_plan['source']['code_sha256']:
        raise ValueError('Parent frozen source checksum mismatch')
    restored = {}
    for case in range(13):
        member = parent/f'X{case:02d}'
        status = member/'status.json'
        if status.exists():
            pid = json.loads(status.read_text()).get('pid')
            if pid and Path(f'/proc/{pid}').exists():
                raise ValueError(f'Parent worker still exists: {pid}')
        checkpoint = member/'checkpoint.npz'
        if not checkpoint.exists():
            continue
        with np.load(checkpoint, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            vector = saved['velocity']
            if metadata['plan_sha256'] != old_hash or metadata['case'] != case or not 0 <= metadata['t'] < FINAL_T:
                raise ValueError('Parent checkpoint identity/time mismatch')
            if hashlib.sha256(vector.tobytes()).hexdigest() != metadata['velocity_sha256']:
                raise ValueError('Parent checkpoint velocity corrupted')
        for name, offset in metadata['offsets'].items():
            if (member/name).stat().st_size < offset:
                raise ValueError('Parent journal is shorter than its committed checkpoint')
        with (member/'fields.jsonl').open('rb') as stream:
            lines = stream.read(metadata['offsets']['fields.jsonl']).decode().splitlines()
        for line in lines:
            field = json.loads(line)
            if digest(member/field['path']) != field['sha256']:
                raise ValueError('Committed parent field checksum mismatch')
        restored[str(case)] = metadata['t']
    shutil.copytree(parent, directory)
    os.replace(directory/'plan.json', directory/'parent-plan.json')
    if (directory/'supervisor.json').exists():
        os.replace(directory/'supervisor.json', directory/'parent-supervisor.json')
    updated = {**old_plan, 'T': FINAL_T, 'source': provenance(),
               'created_utc': datetime.now(timezone.utc).isoformat(),
               'approval': 'User: T改成10000吧，我没有这么多计算时间',
               'parent_plan_sha256': old_hash, 'parent_directory': str(parent),
               'amendment': {'old_T': old_plan['T'], 'new_T': FINAL_T,
                             'restored_checkpoint_times': restored,
                             'scope': 'T and output horizon only; same numerical source except operational production entry'},
               'shutdown_note': 'X0 required SIGKILL after graceful-stop timeout; recovered its last durable checkpoint; others stopped at accepted nodes'}
    atomic_json(directory/'plan.json', updated)
    event('retargeted', old_T=old_plan['T'], new_T=FINAL_T, restored=restored)


def supervise(directory: Path, candidate: Path | None, *, resume: bool = False) -> None:
    """Six-worker lifecycle supervisor, 30-minute process log; no live progress polling."""
    if not resume:
        if candidate is None:
            raise ValueError('An explicit approved candidate is required')
        if shutil.disk_usage(directory.parent).free < 220*1024**3:
            raise RuntimeError('Require at least 220GiB free before starting the approved batch')
        selected = json.loads(candidate.read_text())
        directory.mkdir(exist_ok=False)
        atomic_json(directory/'plan.json', {
            'schema': 'torque_transition_production_v1', 'approved': True,
            'approval': 'User: 不需要测试了，直接执行吧；不需要实时监测进度，留一个log监测任务进程',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'source': provenance(),
            'candidate_sha256': digest(candidate), 'T': FINAL_T, 'grid_cells': 256,
            'nu': 2e-4, 'f0': .1, 'kf': 5, 'gamma': 1000., 'r0': 0.,
            'controller': selected['controller'], 'scalar_atol': 1e-3, 'initial_step': 1e-4,
            'workers': 6, 'threads': 1, 'phase_degrees': PHASE_DEGREES,
            'phase_source': 'inferred_from_rounded_table', 'scalar_interval': 1., 'field_interval': 10.,
            'nearest_node': 'natural accepted node, ties earlier; no interpolation or output truncation',
            'checkpoint': 'atomic accepted state + both PI histories + proposed h + sampler indices + journal offsets',
            'validation_limit': 'User requested no further tests; new streaming/restart path statically checked only; no restart equivalence qualification',
            'production_max_accepted_steps': 100_000_000})
    effective = json.loads((directory/'plan.json').read_text())
    if effective['T'] != FINAL_T or effective['source']['code_sha256'] != provenance()['code_sha256']:
        raise ValueError('Supervisor target/source identity mismatch')
    plan_hash = digest(directory/'plan.json')
    active: dict[int, subprocess.Popen[Any]] = {}
    lock, stopping = RLock(), StopFlag()

    def stop(*_: Any) -> None:
        stopping.set()
        with lock:
            for process in active.values():
                process.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    atomic_json(directory/'supervisor.json', {'pid': os.getpid(), 'status': 'running', 'plan_sha256': plan_hash})
    event('supervisor_started', pid=os.getpid(), directory=str(directory), workers=6, threads=1)

    def launch(case: int) -> dict[str, Any]:
        if stopping.is_set():
            return {'case': case, 'status': 'not_started_stop_requested'}
        status_path = directory/f'X{case:02d}'/'status.json'
        if resume and status_path.exists() and json.loads(status_path.read_text())['status'] == 'complete':
            return {'case': case, 'status': 'already_complete'}
        command = [sys.executable, '-u', '-m', 'experiments.torque_transition.production', 'case',
                   '--output', str(directory), '--case', str(case)]
        if resume and (directory/f'X{case:02d}'/'checkpoint.npz').exists():
            command.append('--resume')
        environment = dict(os.environ, **{key: '1' for key in THREAD_KEYS})
        with (directory/f'X{case:02d}.log').open('a') as log:
            with lock:
                if stopping.is_set():
                    return {'case': case, 'status': 'not_started_stop_requested'}
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=log,
                                           stdin=subprocess.DEVNULL)
                active[case] = process
            event('worker_spawned', case=case, pid=process.pid)
            code = process.wait()
            with lock:
                del active[case]
            event('worker_exited', case=case, pid=process.pid, exit_code=code)
        return {'case': case, 'status': 'complete' if code == 0 else 'failed', 'exit_code': code}

    rows = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        pending = {executor.submit(launch, case) for case in range(13)}
        while pending:
            finished, pending = wait(pending, timeout=1800, return_when=FIRST_COMPLETED)
            for future in finished:
                rows.append(future.result())
            with lock:
                pids = {str(case): process.pid for case, process in active.items()}
            event('process_summary', active=pids, finished=rows,
                  free_disk_gib=shutil.disk_usage(directory).free/1024**3)
    success = all(row['status'] in ('complete', 'already_complete') for row in rows)
    atomic_json(directory/'supervisor.json', {'pid': os.getpid(), 'status': 'complete' if success else 'incomplete',
                'members': rows, 'plan_sha256': plan_hash})
    event('supervisor_finished', success=success)
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('supervise', 'case', 'retarget'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--case', type=int)
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    if arguments.mode == 'supervise':
        supervise(arguments.output, arguments.candidate, resume=arguments.resume)
    elif arguments.mode == 'retarget':
        if arguments.parent is None:
            parser.error('retarget requires --parent')
        retarget(arguments.parent, arguments.output)
    else:
        if arguments.case is None:
            parser.error('case requires --case')
        run_case(arguments.output, arguments.case, resume=arguments.resume)

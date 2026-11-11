"""Opt-in bounded-history execution using the unchanged shared one-step solver.

Only complete, gate-passing segments enter the chain. Each commit links the full
accepted state, checkpoint metadata, parent checkpoint, source and sampling phase.
Failed prefixes are retained outside the committed chain and cannot be resumed.
Wall limits pause at segment boundaries (not a hard per-step time limit).
"""
import json
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Any, Iterator

import h5py
import numpy as np

from experiments.records import (
    PROJECT, begin_run, digest, fail_run, finish_run, provenance, run_identity, save_result, write_json,
)
from solver.core import State
from solver.fourier import FourierSpace, MHDState
from solver.fourier.forced_mhd import (
    ForcedFourierMHD, LOWPASS_SPECTRA, SAMPLED_SPECTRA, transfer_spectra,
)
from solver.integrate import integrate, step_sizes
from solver.schemes.registry import make_scheme, scheme_info
from .chain import CHAIN_SCHEMA_VERSION, chain_digest, load_chain
from .config import effective_config
from .identity import numerical_digest
from .problem import initial_state, paper_force

DEFAULT_SEGMENT_STEPS = 1000
EXECUTION_KEYS = ('segment_steps', 'stop_after_segment', 'resume_from', 'max_wall_seconds')


def _mhd(state: State) -> MHDState:
    if not isinstance(state, MHDState):
        raise TypeError('Expected MHD state')
    return state


def execution_settings(execution: dict[str, Any] | None, total_steps: int) -> dict[str, Any]:
    execution = execution or {}
    if unknown := set(execution)-set(EXECUTION_KEYS):
        raise ValueError(f'Unknown execution keys: {sorted(unknown)}')
    steps = execution.get('segment_steps', DEFAULT_SEGMENT_STEPS)
    if type(steps) is not int or steps <= 0:
        raise ValueError('segment_steps must be a positive integer')
    stop = execution.get('stop_after_segment')
    if stop is not None and (type(stop) is not int or stop <= 0):
        raise ValueError('stop_after_segment must be a positive integer or null')
    wall = execution.get('max_wall_seconds')
    if wall is not None and (isinstance(wall, bool) or not isinstance(wall, (int, float)) or
                             not np.isfinite(wall) or wall <= 0):
        raise ValueError('max_wall_seconds must be positive and finite or null')
    resume = execution.get('resume_from')
    return {'segment_steps': steps, 'stop_after_segment': stop,
            'max_wall_seconds': float(wall) if wall is not None else None,
            'resume_from': str(Path(resume).resolve()) if resume is not None else None,
            'total_steps': int(total_steps), 'planned_segments': int(np.ceil(total_steps/steps))}


def _chain_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory/'chain.json').read_text())


def _segment_path(directory: Path, index: int) -> Path:
    return directory/'segments'/f'{index:04d}.h5'


def _checkpoint_path(directory: Path, index: int) -> Path:
    return directory/'checkpoints'/f'{index:04d}.json'


def _publish(directory: Path, chain: dict, manifest: dict) -> None:
    write_json(directory/'chain.json', chain)
    manifest['chain_sha256'] = digest(directory/'chain.json')
    write_json(directory/'manifest.json', manifest)


@contextmanager
def _writer_lock(directory: Path) -> Iterator[None]:
    lock = directory/'.writer.lock'
    try:
        stream = lock.open('x')
    except FileExistsError as error:
        raise ValueError('Chain writer lock exists; verify the old process before manual recovery') from error
    try:
        stream.close()
        yield
    finally:
        lock.unlink()


def _model(c: dict) -> ForcedFourierMHD:
    space = FourierSpace(c['nx'], c['ny'])
    forcing = paper_force(space, c['kf'], c['f0'], c['mu'])
    return ForcedFourierMHD(space, c['nu_plus'], c['nu_minus'], force=lambda t: forcing)


def run_segmented(config: dict, *, execution: dict[str, Any] | None = None,
                  root: Path = PROJECT) -> Path:
    """Explicit new chain or explicit resume; no implicit lookup of latest state."""
    c = effective_config(config)
    settings = execution_settings(execution, round(c['T']/c['dt']))
    source = provenance()
    identity = run_identity(c, source)
    numerical = numerical_digest()
    if settings['resume_from'] is not None:
        directory = Path(settings['resume_from'])
        with _writer_lock(directory):
            loaded = load_chain(directory, assemble=False)
            chain, manifest = loaded['chain'], loaded['manifest']
            if loaded['config'] != c or chain['total_steps'] != settings['total_steps']:
                raise ValueError('Cannot resume: requested configuration differs from the chain')
            if chain['status'] == 'complete':
                raise ValueError('Cannot resume: chain already reached its planned terminal time')
            if chain.get('chain_schema_version', 1) != CHAIN_SCHEMA_VERSION:
                raise ValueError('Legacy chain is read-only; current code cannot resume its historical source')
            if 'numerical_sha256' in chain:
                if chain['numerical_sha256'] != numerical:
                    raise ValueError('Cannot resume: numerical source changed since this chain was committed')
            elif chain['identity'] != identity:
                # Pre-split chains only carried the aggregate hash; migrate explicitly.
                raise ValueError('Cannot resume: chain has no numerical digest; run migrate_identity() first')
            if execution and 'segment_steps' in execution and \
                    settings['segment_steps'] != chain['execution']['segment_steps']:
                raise ValueError('Cannot change segment_steps on an existing chain')
            _execute(directory, c, settings, manifest, loaded['final'])
    else:
        directory, manifest = begin_run(root/'runs'/'inverse_cascade_chains', c, source,
                                        metadata={'experiment': 'inverse_cascade_chain'})
        write_json(directory/'execution.json', settings)
        (directory/'segments').mkdir()
        (directory/'checkpoints').mkdir()
        chain = {'chain_schema_version': CHAIN_SCHEMA_VERSION, 'status': 'running',
                 'identity': identity, 'numerical_sha256': numerical,
                 'config_sha256': manifest['config_sha256'],
                 'source': source, 'segments': [], 'accepted_steps': 0, 'sample_count': 0,
                 'total_steps': settings['total_steps'], 'error': '', 'stop_reason': '',
                 'execution': {k: v for k, v in settings.items() if k != 'resume_from'}}
        _publish(directory, chain, manifest)
        with _writer_lock(directory):
            _execute(directory, c, settings, manifest, None)
    return directory


def _execute(directory: Path, c: dict, settings: dict, manifest: dict,
             restart: dict[str, Any] | None) -> None:
    chain = _chain_manifest(directory)
    metrics: dict[str, Any] = dict(manifest.get('metrics', {}))
    started = monotonic()
    produced = 0
    try:
        if restart is None:
            state: State = initial_state(_model(c), c['kf'], c['f0'], c['mu'], c['seed'], c['noise_rms'])
        else:
            state = MHDState(float(restart['t']), restart['u'], restart['v'], float(restart['r']),
                             magnetic=restart['magnetic'])
        while chain['accepted_steps'] < chain['total_steps']:
            if settings['stop_after_segment'] is not None and produced >= settings['stop_after_segment']:
                chain['stop_reason'] = 'stop_after_segment'
                break
            if settings['max_wall_seconds'] is not None and monotonic()-started >= settings['max_wall_seconds']:
                chain['stop_reason'] = 'max_wall_seconds'
                break
            steps = min(chain['execution']['segment_steps'], chain['total_steps']-chain['accepted_steps'])
            state = _advance(directory, state, steps, c, chain, metrics)
            produced += 1
            manifest['metrics'] = metrics
            _publish(directory, chain, manifest)
        chain['stop_reason'] = chain['stop_reason'] or 'segment_boundary'
        if chain['accepted_steps'] == chain['total_steps']:
            chain.update(status='complete', stop_reason='terminal_time')
            _publish(directory, chain, manifest)
            finish_run(directory, manifest, status='complete', error='', metrics=metrics,
                       results_sha256=chain_digest(chain))
        else:
            _publish(directory, chain, manifest)
    except Exception as error:
        chain.update(status='failed', error=f'{type(error).__name__}: {error}')
        _publish(directory, chain, manifest)
        fail_run(directory, manifest, error)
        with (directory/'run.log').open('a') as stream:
            stream.write(chain['error']+'\n')
        raise


def _advance(directory: Path, state: State, steps: int, c: dict, chain: dict,
             metrics: dict[str, Any]) -> State:
    model = _model(c)
    index = len(chain['segments'])
    before = chain['accepted_steps']
    samples: list[dict[str, Any]] = []
    if index == 0:
        samples.append({'time': state.t, **transfer_spectra(model, model.vector(_mhd(state)), state.t)})
    counter = before

    def sample(current: State) -> None:
        nonlocal counter
        counter += 1
        if counter % c['sample_every'] == 0:
            samples.append({'time': current.t, **transfer_spectra(model, model.vector(_mhd(current)), current.t)})

    dt = c['dt']*c['t0']
    result = integrate(model, make_scheme(c['scheme'], c['gamma']/c['t0']), state,
                       step_sizes(steps*dt, dt=dt), snapshots=[], on_accept=sample)
    advanced = len(result.times)-1
    deviation = 0.
    if c['scheme'].endswith('mrsav') and result.stages:
        deviation = float(np.max(np.abs([s['frozen_factor']*(1-s['r'])-1 for row in result.stages for s in row])))
    if not np.isfinite(deviation):
        result.status, result.error = 'failed', 'Nonfinite SAV coupling factor'
    elif c.get('coupling_tolerance') is not None and deviation > c['coupling_tolerance']:
        result.status, result.error = 'failed', f'SAV coupling deviation {deviation:.3e} exceeds coupling_tolerance'
    if advanced != steps:
        result.status, result.error = 'failed', result.error or f'Segment advanced {advanced} of {steps} steps'
    step_metrics = {'accepted_steps': advanced, 'integration_wall_seconds': result.seconds,
                    'observer_wall_seconds': result.observer_wall_seconds,
                    'max_abs_coupling_minus_one': deviation if np.isfinite(deviation) else None}
    path = _segment_path(directory, index) if result.status == 'complete' else directory/'failed'/f'{index:04d}.h5'
    pending = path.with_suffix('.pending.h5')
    path.parent.mkdir(exist_ok=True)
    checkpoint_path = _checkpoint_path(directory, index)
    if path.exists() or pending.exists() or checkpoint_path.exists():
        raise ValueError('Uncommitted payload exists; inspect/quarantine it explicitly before recovery')
    info = scheme_info(c['scheme'])
    save_result(pending, result, step_metrics, n_stages=info.n_stages, n_roots=info.n_roots,
                pressure_semantics=c['pressure_semantics'], include_stage_pressure=False)
    final = _mhd(result.final)
    K = np.arange(int(np.ceil(np.hypot(c['retained_mx'][1], c['retained_my'][1])))+1, dtype=float)
    with h5py.File(pending, 'a') as d:
        for key, value in (('u', state.u), ('v', state.v), ('r', state.r), ('t', state.t),
                           ('magnetic', _mhd(state).magnetic)):
            d.create_dataset('initial/'+key, data=value)
        d.create_dataset('final/magnetic', data=final.magnetic)
        d.create_dataset('spectra/K', data=K)
        d.create_dataset('spectra/times', data=np.array([v['time'] for v in samples], dtype=float))
        for key in SAMPLED_SPECTRA+LOWPASS_SPECTRA:
            d.create_dataset('spectra/'+key, data=np.array([v[key] for v in samples]).reshape(-1, len(K)),
                             compression='gzip')
        d.attrs['segment_index'] = index
        d.attrs['chain_identity'] = chain['identity']
    pending.replace(path)
    if result.status != 'complete':
        raise RuntimeError(result.error or 'segment did not complete')
    total_samples = chain['sample_count']+len(samples)
    parent = chain['segments'][-1]['checkpoint_sha256'] if chain['segments'] else None
    checkpoint = {'segment': index, 'steps': steps, 'steps_before': before, 'steps_after': counter,
                  'sample_count': total_samples, 'sample_phase': counter % c['sample_every'], 't': float(final.t),
                  'config_sha256': chain['config_sha256'], 'segment_sha256': digest(path),
                  'chain_identity': chain['identity'], 'numerical_sha256': chain.get('numerical_sha256'),
                  'harness_sha256': provenance()['code_sha256'],
                  'parent_checkpoint_sha256': parent, 'status': 'complete'}
    write_json(checkpoint_path, checkpoint)
    chain['segments'].append({'index': index, 'file': f'segments/{index:04d}.h5', 'steps': steps,
                              'steps_after': counter, 't_start': float(state.t), 't_end': float(final.t),
                              'sha256': checkpoint['segment_sha256'], 'checkpoint_sha256': digest(checkpoint_path),
                              'status': 'complete', 'samples': len(samples), 'max_abs_coupling_minus_one': deviation})
    chain.update(accepted_steps=counter, sample_count=total_samples, stop_reason='')
    metrics.update(segments=len(chain['segments']), accepted_steps=counter,
                   integration_wall_seconds=metrics.get('integration_wall_seconds', 0.)+result.seconds,
                   observer_wall_seconds=metrics.get('observer_wall_seconds', 0.)+result.observer_wall_seconds)
    with (directory/'run.log').open('a') as stream:
        stream.write(f'segment={index} committed steps={steps} samples={len(samples)} wall={result.seconds:.3f}s\n')
    return final


def migrate_identity(directory: Path, *, reason: str, reviewer: str) -> dict[str, Any]:
    """One-time, audited split of a pre-existing chain identity into numerical+harness.

    Chains created before the numerical/harness split only carried one aggregate
    source hash, so a reader-only fix blocked continuation. This refuses to run
    unless every committed segment validates first, refuses a second migration, and
    records the old identity, the new numerical digest, the current harness digest
    and the stated reason. It never rewrites segment payloads or checkpoints.
    """
    if not reason.strip() or not reviewer.strip():
        raise ValueError('Migration needs an explicit reason and named reviewer')
    directory = Path(directory)
    with _writer_lock(directory):
        loaded = load_chain(directory, assemble=False)
        chain, manifest = loaded['chain'], loaded['manifest']
        if 'numerical_sha256' in chain:
            raise ValueError('Chain already carries a numerical digest')
        if chain['status'] != 'running':
            raise ValueError('Only a running chain can be migrated')
        numerical = numerical_digest()
        current = provenance()['code_sha256']
        chain['numerical_sha256'] = numerical
        chain['identity_migration'] = [{
            'from_aggregate_sha256': chain['identity'], 'to_numerical_sha256': numerical,
            'harness_sha256_at_migration': current, 'reviewer': reviewer.strip(),
            'reason': reason.strip(), 'validated_segments': len(chain['segments']),
            'validated_steps': chain['accepted_steps'],
            'does_not_change': ['segment payloads', 'checkpoints', 'accepted trajectory']}]
        manifest.setdefault('harness_history', []).append(current)
        _publish(directory, chain, manifest)
    with (directory/'run.log').open('a') as stream:
        stream.write(f'migration numerical={numerical} harness={current} reason={reason.strip()}\n')
    return {'numerical_sha256': numerical, 'harness_sha256': current,
            'validated_steps': chain['accepted_steps']}

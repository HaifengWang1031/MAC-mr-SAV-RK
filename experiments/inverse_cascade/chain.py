"""Read-only integrity gate for chains; validation is also used before resume.

New chains bind checkpoints, parent links and chain metadata by checksums. Legacy
schema-1 chains can be read (with weaker historical metadata guarantees) but cannot
be resumed by current execution. ``assemble=False`` validates one segment at a
time without accumulating histories, so restart memory does not grow with T.
"""
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from experiments.records import digest, run_identity
from solver.fourier.forced_mhd import LOWPASS_SPECTRA, SAMPLED_SPECTRA
from solver.schemes.registry import scheme_info
from .config import CASE, DEFAULTS, OPTIONAL, effective_config

DIAGNOSTICS = ('total_energy', 'potential_squared', 'power', 'power_Q',
               'dissipation_E_small', 'dissipation_E_large',
               'dissipation_Q_small', 'dissipation_Q_large', 'r')
STAGES = ('r', 'residual', 'scalar_residual', 'divergence_inf', 'continuity_residual')
SPECTRA = SAMPLED_SPECTRA+LOWPASS_SPECTRA
CHAIN_SCHEMA_VERSION = 2


def chain_digest(chain: dict[str, Any]) -> str:
    """Ordered digest of immutable payloads, including checkpoint metadata for v2."""
    entries: list[tuple[Any, ...]]
    if chain.get('chain_schema_version', 1) >= 2:
        entries = [(s['index'], s['sha256'], s['checkpoint_sha256']) for s in chain['segments']]
    else:
        entries = [(s['index'], s['sha256']) for s in chain['segments']]
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def _equal_time(actual: float, expected: float) -> bool:
    """Compare step-derived times across unavoidable double-precision accumulation.

    The integrator advances ``t += h``. After 2e5 steps the accumulated time differs
    from the ideal ``n*h`` by a few 1e-12 relative (measured 4.4e-10 absolute at
    t=186), while a missing/duplicated step would differ by one ``h`` (~5e-6
    relative). The tolerance therefore has to sit well above accumulation noise and
    far below one step; 1e-9 relative / 1e-12 absolute does that.
    """
    return bool(np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-9,
                                                  atol=1e-12*max(1., abs(expected))))


def _header(directory: Path, require_complete: bool) -> tuple[dict, dict, dict]:
    c = json.loads((directory/'config.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    chain = json.loads((directory/'chain.json').read_text())
    if c.get('case') != CASE or manifest.get('experiment') != 'inverse_cascade_chain':
        raise ValueError('Expected an inverse_cascade chain')
    version = chain.get('chain_schema_version', 1)
    if version not in (1, CHAIN_SCHEMA_VERSION):
        raise ValueError('Unsupported chain schema')
    if 'failed' in (chain.get('status'), manifest.get('status')):
        raise ValueError('A failed chain cannot be analyzed or resumed')
    if chain.get('status') != manifest.get('status') or chain.get('status') not in ('running', 'complete'):
        raise ValueError('Chain/manifest status mismatch')
    if require_complete and chain['status'] != 'complete':
        raise ValueError('Chain has not reached its planned terminal time')
    checksum = digest(directory/'config.json')
    if checksum != chain.get('config_sha256') or checksum != manifest.get('config_sha256'):
        raise ValueError('Chain config checksum mismatch')
    if chain.get('source') != manifest.get('source') or \
            chain.get('identity') != manifest.get('identity') or \
            run_identity(c, chain['source']) != chain['identity']:
        raise ValueError('Chain source/config identity mismatch')
    raw = {key: c[key] for key in (*DEFAULTS, *OPTIONAL) if key in c}
    if effective_config(raw) != c:
        raise ValueError('Invalid effective chain configuration/layout')
    if version >= 2 and manifest.get('chain_sha256') != digest(directory/'chain.json'):
        raise ValueError('Chain metadata checksum mismatch')
    if chain['total_steps'] != round(c['T']/c['dt']):
        raise ValueError('Chain terminal step count mismatch')
    if type(chain['execution'].get('segment_steps')) is not int or chain['execution']['segment_steps'] <= 0:
        raise ValueError('Invalid segment length')
    if chain['status'] == 'complete' and manifest.get('results_sha256') != chain_digest(chain):
        raise ValueError('Completed chain payload digest mismatch')
    return c, manifest, chain


def _finite(d: h5py.File, key: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(d[key][()])
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f'Invalid finite shape: {key}')
    return values


def _state(d: h5py.File, group: str, c: dict) -> dict[str, Any]:
    shapes = {'u': (c['ny'], c['nx']), 'v': (c['ny'], c['nx']),
              'magnetic': (2, c['ny'], c['nx']), 'r': (), 't': ()}
    return {key: _finite(d, group+'/'+key, shape) for key, shape in shapes.items()}


def load_chain(directory: Path, *, require_complete: bool = False,
               assemble: bool = True) -> dict[str, Any]:
    """Validate all committed payloads, not only the final checkpoint.

    Failed chains are rejected even when ``require_complete=False``. Intentional
    segment-boundary pauses remain running and can be resumed after validation.
    """
    c, manifest, chain = _header(directory, require_complete)
    segments = chain['segments']
    version = chain.get('chain_schema_version', 1)
    if [s['index'] for s in segments] != list(range(len(segments))):
        raise ValueError('Chain segment indices are not contiguous from zero')
    parts: dict[str, list[np.ndarray]] = {}
    previous: dict[str, Any] | None = None
    previous_diagnostics: dict[str, float] | None = None
    previous_steps, sample_count = 0, 0
    parent: str | None = None
    info = scheme_info(c['scheme'])
    K = np.arange(int(np.ceil(np.hypot(c['retained_mx'][1], c['retained_my'][1])))+1, dtype=float)

    def append(key: str, value: np.ndarray) -> None:
        if assemble:
            parts.setdefault(key, []).append(value)

    for index, segment in enumerate(segments):
        expected_file = f'segments/{index:04d}.h5'
        if segment['file'] != expected_file:
            raise ValueError('Unexpected segment path')
        path = directory/expected_file
        if digest(path) != segment['sha256']:
            raise ValueError('Segment checksum mismatch')
        checkpoint_path = directory/'checkpoints'/f'{index:04d}.json'
        checkpoint_hash = digest(checkpoint_path)
        if version >= 2 and checkpoint_hash != segment.get('checkpoint_sha256'):
            raise ValueError('Checkpoint metadata checksum mismatch')
        checkpoint = json.loads(checkpoint_path.read_text())
        steps = segment['steps']
        if type(steps) is not int or steps <= 0 or steps > chain['execution']['segment_steps']:
            raise ValueError('Invalid segment step count')
        after = previous_steps+steps
        if after > chain['total_steps']:
            raise ValueError('Segment exceeds planned terminal step')
        expected_samples = after//c['sample_every']-previous_steps//c['sample_every']+(index == 0)
        sample_count += expected_samples
        for key, expected in (('segment', index), ('steps', steps), ('steps_after', after),
                              ('sample_count', sample_count), ('config_sha256', chain['config_sha256']),
                              ('segment_sha256', segment['sha256']), ('status', 'complete')):
            if checkpoint.get(key) != expected:
                raise ValueError(f'Checkpoint {key} mismatch')
        if segment['steps_after'] != after or segment['samples'] != expected_samples or \
                segment['status'] != 'complete':
            raise ValueError('Segment step/sample/status mismatch')
        if version >= 2:
            for key, expected in (('chain_identity', chain['identity']), ('parent_checkpoint_sha256', parent),
                                  ('sample_phase', after % c['sample_every']), ('steps_before', previous_steps)):
                if checkpoint.get(key) != expected:
                    raise ValueError(f'Checkpoint {key} mismatch')
        with h5py.File(path, 'r') as d:
            if d.attrs['status'] != 'complete' or d.attrs['segment_index'] != index or \
                    d.attrs['chain_identity'] != chain['identity'] or \
                    d.attrs['pressure_semantics'] != c['pressure_semantics']:
                raise ValueError('Segment attributes/status mismatch')
            times = _finite(d, 'times', (steps+1,))
            if np.any(np.diff(times) <= 0) or \
                    not np.allclose(np.diff(times), c['dt']*c['t0'], rtol=1e-8, atol=1e-12) or \
                    not _equal_time(float(times[0]), previous_steps*c['dt']*c['t0']) or \
                    not _equal_time(float(times[-1]), after*c['dt']*c['t0']):
                raise ValueError('Segment time continuity/step mismatch')
            if not _equal_time(segment['t_start'], float(times[0])) or \
                    not _equal_time(segment['t_end'], float(times[-1])) or \
                    not _equal_time(checkpoint['t'], float(times[-1])):
                raise ValueError('Segment/checkpoint time mismatch')
            drop = int(index > 0)
            append('times', times[drop:])
            current_diagnostics = {}
            for key in DIAGNOSTICS:
                data = _finite(d, 'diagnostics/'+key, (steps+1,))
                if previous_diagnostics is not None and not np.isclose(
                        data[0], previous_diagnostics[key], rtol=1e-12, atol=1e-14):
                    raise ValueError('Diagnostic continuity mismatch')
                append('diagnostics/'+key, data[drop:])
                current_diagnostics[key] = float(data[-1])
            previous_diagnostics = current_diagnostics
            for key in STAGES:
                append('stages/'+key, _finite(d, 'stages/'+key, (steps, info.n_stages)))
            counts = _finite(d, 'corrections/count', (steps, info.n_stages))
            if np.any(counts != info.n_corrections):
                raise ValueError('Invalid scalar correction counts')
            factors = np.asarray(d['corrections/frozen_factor'][:])
            stage_r = np.asarray(d['stages/r'][:])
            if factors.shape != (steps, info.n_stages):
                raise ValueError('Invalid coupling shape')
            if info.n_corrections:
                values = _finite(d, 'corrections/values', (steps, info.n_stages, info.n_corrections))
                if not np.array_equal(values[..., -1], stage_r) or \
                        not np.allclose(factors, 1+values[..., 0]+values[..., 0]**2, rtol=1e-12, atol=1e-14):
                    raise ValueError('Invalid scalar correction values/factor')
                deviation = float(np.max(np.abs(factors*(1-stage_r)-1)))
                if not np.isfinite(deviation) or c.get('coupling_tolerance') is not None and \
                        deviation > c['coupling_tolerance']:
                    raise ValueError('Segment coupling gate failed')
            else:
                deviation = 0.
            if not np.isclose(segment['max_abs_coupling_minus_one'], deviation, rtol=1e-12, atol=1e-14):
                raise ValueError('Segment coupling metric mismatch')
            append('corrections/frozen_factor', factors)
            sample_times = _finite(d, 'spectra/times', (expected_samples,))
            selected_steps = np.r_[0 if index == 0 else np.empty(0, dtype=int),
                                   np.arange(previous_steps+1, after+1)[
                                       np.arange(previous_steps+1, after+1) % c['sample_every'] == 0]].astype(int)
            if not np.allclose(sample_times, times[selected_steps-previous_steps], rtol=0, atol=1e-10):
                raise ValueError('Segment spectral sampling phase mismatch')
            if not np.array_equal(_finite(d, 'spectra/K', K.shape), K):
                raise ValueError('Invalid radial cutoffs')
            append('spectra/times', sample_times)
            for key in SPECTRA:
                append('spectra/'+key, _finite(d, 'spectra/'+key, (expected_samples, len(K))))
            if version >= 2:
                initial = _state(d, 'initial', c)
                if previous is not None and any(not np.array_equal(initial[k], previous[k]) for k in initial):
                    raise ValueError('Full accepted-state continuity mismatch')
                if not _equal_time(float(initial['t']), float(times[0])):
                    raise ValueError('Initial-state time mismatch')
            previous = _state(d, 'final', c)
            if not _equal_time(float(previous['t']), float(times[-1])) or \
                    float(previous['r']) != float(stage_r[-1, -1]):
                raise ValueError('Final state/time/scalar mismatch')
        previous_steps, parent = after, checkpoint_hash
    if chain['accepted_steps'] != previous_steps or chain['sample_count'] != sample_count:
        raise ValueError('Chain accepted-step/sample totals mismatch')
    if chain['status'] == 'complete' and previous_steps != chain['total_steps']:
        raise ValueError('Chain does not cover its planned terminal time')
    result: dict[str, Any] = {'config': c, 'manifest': manifest, 'chain': chain, 'final': previous,
                              'K': K, 'arrays': None, 'sample_times': None}
    if assemble:
        if not segments:
            raise ValueError('Chain has no committed segments')
        flat = {key: np.concatenate(value) for key, value in parts.items()}
        result['arrays'] = {'times': flat['times'],
                            'stages': {key: flat['stages/'+key] for key in STAGES},
                            'corrections': flat['corrections/frozen_factor'],
                            'spectra': {key: flat['spectra/'+key] for key in SPECTRA},
                            'diagnostics': {key: flat['diagnostics/'+key] for key in DIAGNOSTICS}}
        result['sample_times'] = flat['spectra/times']
    return result


def chain_source(directory: Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Read-only adapter exposing the same dataset mapping as single-run HDF5."""
    loaded = load_chain(directory, require_complete=require_complete)

    def opener() -> Any:
        arrays = loaded['arrays']
        flat: dict[str, Any] = {'times': arrays['times'], 'spectra/K': loaded['K'],
                                'spectra/times': loaded['sample_times']}
        for group in ('diagnostics', 'stages', 'spectra'):
            for key, value in arrays[group].items():
                flat[group+'/'+key] = value
        flat['corrections/frozen_factor'] = arrays['corrections']
        return nullcontext(flat)

    return {'name': str(directory.resolve()), 'c': loaded['config'],
            'manifest': loaded['manifest'], 'open': opener}

"""Run identity, persistence and integrity without importing computation entry points."""
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4
import hashlib
import importlib.metadata
import json
import os
import platform
import sys

import h5py
import numpy as np

if TYPE_CHECKING:
    from solver.integrate import Result

PROJECT = Path(__file__).resolve().parents[1]


def run_identity(config: dict[str, Any], source: dict[str, Any]) -> str:
    """Keep the historical canonical config/source identity encoding."""
    return hashlib.sha256(json.dumps({'config': config, 'source': source}, sort_keys=True).encode()).hexdigest()


def reusable_run(parent: Path, config: dict[str, Any], source: dict[str, Any], *,
                 reader: Callable[[Path], tuple[Any, ...]], rerun: bool = False) -> Path | None:
    """Find a complete same-config/source run through its representation reader."""
    if rerun or not parent.exists():
        return None
    identity = run_identity(config, source)
    for directory in sorted(parent.iterdir()):
        if not directory.is_dir():
            continue
        try:
            old, manifest, *_ = reader(directory)
            if manifest.get('status') == 'complete' and old == config \
                    and manifest.get('identity') == identity and manifest.get('source') == source:
                return directory
        except (OSError, KeyError, ValueError):
            continue
    return None


def begin_run(parent: Path, config: dict[str, Any], source: dict[str, Any], *,
              metadata: dict[str, Any], started_key: str = 'started') -> tuple[Path, dict[str, Any]]:
    """Create a new attempt and persist its config before publishing its manifest."""
    identity = run_identity(config, source)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    directory = parent/f'{stamp}-{identity[:8]}-{uuid4().hex[:8]}'
    directory.mkdir()
    manifest = {'run_id': directory.name, 'identity': identity, 'source': source,
                'status': 'running', started_key: datetime.now(timezone.utc).isoformat(), **metadata}
    write_json(directory/'config.json', config)
    manifest['config_sha256'] = digest(directory/'config.json')
    write_json(directory/'manifest.json', manifest)
    return directory, manifest


def finish_run(directory: Path, manifest: dict[str, Any], *, status: str, error: str,
               metrics: dict[str, Any], finished_key: str = 'finished',
               fields: dict[str, Any] | None = None, results_sha256: str | None = None) -> None:
    """Finalize only after the experiment has published its complete result file."""
    checksum = digest(directory/'results.h5') if results_sha256 is None else results_sha256
    manifest.update(status=status, error=error, metrics=metrics, results_sha256=checksum,
                    **{finished_key: datetime.now(timezone.utc).isoformat()}, **(fields or {}))
    write_json(directory/'manifest.json', manifest)


def fail_run(directory: Path, manifest: dict[str, Any], error: Exception, *,
             finished_key: str = 'finished') -> None:
    """Retain an auditable setup/storage failure even when no result is available."""
    manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                    **{finished_key: datetime.now(timezone.utc).isoformat()})
    write_json(directory/'manifest.json', manifest)

def write_json(path: Path, data: dict) -> None:
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)

def digest(path: Path) -> str:
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): hasher.update(chunk)
    return hasher.hexdigest()

def provenance() -> dict:
    hasher=hashlib.sha256()
    files=[PROJECT/'pyproject.toml',PROJECT/'uv.lock',PROJECT/'.python-version']
    for directory in ('solver','experiments','tools'):
        files.extend(sorted((PROJECT/directory).rglob('*.py')))
    for path in files:
        hasher.update(str(path.relative_to(PROJECT)).encode()); hasher.update(path.read_bytes())
    return {'code_sha256':hasher.hexdigest(),'python':sys.version,'platform':platform.platform(),
            'packages':{name:importlib.metadata.version(name) for name in ('numpy','scipy','numba','h5py','sympy','matplotlib')},
            'threads':{key:os.environ.get(key) for key in (
                'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')}}

def save_scalar_corrections(data: Any, stages: list[list[dict]], n_stages: int) -> None:
    """Shared additive scalar diagnostics; legacy root slots retain their original layout."""
    selected = np.asarray([[stage['r'] for stage in row] for row in stages]).reshape(-1, n_stages)
    data.create_dataset('stages/r', data=selected)
    capacity = 2 if n_stages == 4 else 1
    correction_counts = np.zeros((len(stages), n_stages), dtype=int)
    correction_values = np.full((*correction_counts.shape, capacity), np.nan)
    predictors = np.full(correction_counts.shape, np.nan)
    factors = predictors.copy()
    for n, row in enumerate(stages):
        for i, stage in enumerate(row):
            values = stage.get('corrections', [])
            correction_counts[n, i] = len(values)
            correction_values[n, i, :len(values)] = values
            predictors[n, i] = stage.get('correction_predictor', np.nan)
            factors[n, i] = stage.get('frozen_factor', np.nan)
    data.attrs['scalar_solver'] = ('fixed_count_frozen_factor' if np.any(correction_counts)
                                  else 'legacy_polynomial_roots' if any(s.get('root_count', 0) for row in stages for s in row) else 'none')
    for key, value in [('count', correction_counts), ('values', correction_values),
                       ('predictor', predictors), ('frozen_factor', factors)]:
        data.create_dataset('corrections/'+key, data=value)

def save_result(path: Path, result: "Result", metrics: dict, n_stages: int = 2, *,
                n_roots: int | None = None, pressure_semantics: str | None = None) -> None:
    if n_roots is None: n_roots = {1: 1, 2: 3, 4: 5}[n_stages]
    temporary=path.with_suffix('.tmp')
    with h5py.File(temporary,'w') as data:
        data.attrs['schema_version']=1
        data.attrs['status']=result.status
        data.attrs['metrics_json']=json.dumps(metrics,allow_nan=False)
        if pressure_semantics is not None: data.attrs['pressure_semantics'] = pressure_semantics
        data.create_dataset('times',data=result.times)
        data.create_dataset('dt',data=np.diff(result.times))
        for key in result.diagnostics[0]:
            data.create_dataset('diagnostics/'+key,data=[item[key] for item in result.diagnostics])
        for key,value in [('u',result.final.u),('v',result.final.v),('r',result.final.r),('t',result.final.t)]:
            data.create_dataset('final/'+key,data=value)
        if result.final_stages:
            data.create_dataset('final/stage_pressure',data=np.stack([s.pressure for s in result.final_stages]))
        data.create_dataset('snapshots/requested_times',data=result.snapshot_requests)
        data.create_dataset('snapshots/actual_times',data=result.snapshot_times)
        if result.snapshots:
            data.create_dataset('snapshots/u',data=np.stack([s.u for s in result.snapshots]),compression='gzip')
            data.create_dataset('snapshots/v',data=np.stack([s.v for s in result.snapshots]),compression='gzip')
            data.create_dataset('snapshots/r',data=[s.r for s in result.snapshots])
        candidates=np.full((len(result.stages),n_stages,n_roots),np.nan)
        residuals=candidates.copy()
        counts=np.zeros((len(result.stages),n_stages),dtype=int)
        selected=np.zeros_like(counts,dtype=float)
        for n,stages in enumerate(result.stages):
            for i,stage in enumerate(stages):
                count=stage['root_count']; counts[n,i]=count; selected[n,i]=stage['r']
                candidates[n,i,:count]=stage['candidates']; residuals[n,i,:count]=stage['root_residuals']
        for key,value in [('candidates',candidates),('residuals',residuals),('count',counts),('selected',selected)]:
            data.create_dataset('roots/'+key,data=value)
        save_scalar_corrections(data, result.stages, n_stages)
        for key in ('residual','scalar_residual','divergence_inf','continuity_residual'):
            data.create_dataset('stages/'+key,data=np.array([[s[key] for s in stages] for stages in result.stages]).reshape(-1,n_stages))
    temporary.replace(path)

def load_record(directory: Path, *, require_complete: bool = False) -> tuple[dict,dict]:
    """Shared integrity gate for reuse and every analysis reader."""
    manifest=json.loads((directory/'manifest.json').read_text())
    if digest(directory/'results.h5')!=manifest.get('results_sha256') or digest(directory/'config.json')!=manifest.get('config_sha256'):
        raise ValueError(f'Record checksum mismatch: {directory}')
    if manifest.get('status') not in ('complete','failed'):
        raise ValueError('Record is not finalized')
    if require_complete and manifest['status']!='complete': raise ValueError('Complete run required')
    return json.loads((directory/'config.json').read_text()),manifest

def verified(manifest: dict, directory: Path) -> bool:
    try:
        load_record(directory,require_complete=True)
        return True
    except (OSError,KeyError,ValueError): return False

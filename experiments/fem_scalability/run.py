"""Prepare frozen fixtures and run serial, interleaved comparison workers.

Examples: python -m experiments.fem_scalability.run prepare --run DIR
          python -m experiments.fem_scalability.run measure --run DIR --levels 0 --repeats 3
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
THREAD_ENV = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                  'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')}
VARIANTS = {
    'original_direct': ('baseline_source', {'stokes_backend': 'direct'}),
    'original_recycled': ('baseline_source', {'stokes_backend': 'recycled'}),
    'mass_cg': ('candidate_source', {'stokes_backend': 'recycled', 'mass_backend': 'cg'}),
    'weak_recycled': ('candidate_source', {'stokes_backend': 'recycled', 'fem_load_mode': 'weak'}),
    'weak_recycled_cg': ('candidate_source', {'stokes_backend': 'recycled', 'fem_load_mode': 'weak', 'mass_backend': 'cg'}),
    'metis_recycled': ('candidate_v2_source', {'stokes_backend': 'recycled', 'mass_backend': 'metis'}),
    'weak_metis_recycled': ('candidate_v2_source', {'stokes_backend': 'recycled', 'mass_backend': 'metis', 'fem_load_mode': 'weak'}),
    'refined_amg_cg': ('candidate_v2_source', {'stokes_backend': 'petsc', 'fem_load_mode': 'weak', 'mass_backend': 'cg'}),
    'weak_amg': ('candidate_source', {'stokes_backend': 'petsc', 'fem_load_mode': 'weak'}),
    'weak_amg_cg': ('candidate_source', {'stokes_backend': 'petsc', 'fem_load_mode': 'weak', 'mass_backend': 'cg'}),
}


def write(path: Path, value: Any) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def freeze(source: Path, target: Path) -> dict[str, str]:
    target.mkdir()
    paths = [p for name in ('solver', 'experiments', 'tools') for p in (source/name).rglob('*.py')]
    paths += [source/name for name in ('pyproject.toml', 'uv.lock', '.python-version')]
    result = {}
    for path in paths:
        if path.is_file():
            relative = path.relative_to(source)
            output = target/relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            result[str(relative)] = hashlib.sha256(output.read_bytes()).hexdigest()
    return result


def invoke(run: Path, fixture: Path, output: Path, source: str, workload: str,
           options: dict[str, Any], scheme: str, timeout: float = 300.) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(run/('worker_v2' if source=='candidate_v2_source' and (run/'worker_v2').exists() else
                     'worker_adaptive_v2' if workload=='adaptive' and (run/'worker_adaptive_v2').exists() else
                     'worker_step' if workload=='step' and (run/'worker_step').exists() else 'worker')/'_worker.py'), '--source', str(run/source),
        '--fixture', str(fixture), '--output', str(output), '--workload', workload,
        '--options', json.dumps(options), '--scheme', scheme]
    started = perf_counter()
    try:
        with output.with_suffix('.log').open('w') as log:
            completed = subprocess.run(command, cwd=run, env={**os.environ, **THREAD_ENV},
                stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        result = json.loads(output.read_text()) if output.exists() else {'status': 'failed', 'error': 'no worker result'}
        result.update(exit_code=completed.returncode)
        if completed.returncode and result.get('status') in ('running', 'complete'):
            result.update(status='failed', error='Worker process exited before completing all phases')
    except subprocess.TimeoutExpired:
        result = json.loads(output.read_text()) if output.exists() else {}
        result.update(status='timeout', error=f'worker exceeded {timeout:g}s', limit_seconds=timeout)
    if output.with_suffix('.npz').exists():
        result['field_sha256'] = hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest()
    result.update(end_to_end_seconds=perf_counter()-started,
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(output, result)
    return result


def prepare(run: Path) -> None:
    baseline = run/'baseline_source'
    if not baseline.is_dir():
        raise ValueError('Expected the pre-edit baseline_source in the explicit run directory')
    if (run/'campaign.json').exists():
        raise ValueError('Campaign already prepared; select its existing fixtures for measurement')
    hashes = freeze(PROJECT, run/'candidate_source')
    shutil.copytree(Path(__file__).parent, run/'worker', ignore=shutil.ignore_patterns('__pycache__'))
    fixtures = []
    for family in ('cavity', 'cylinder', 'wedge', 'hele'):
        for level in range(3):
            spec: dict[str, Any] = dict(family=family, level=level, quadrature_order=6)
            if family == 'cavity':spec['n'] = 32*2**level
            if family == 'wedge':spec['n'] = 32*2**level
            if family == 'cylinder':spec['resolution'] = .5*2**level
            if family == 'hele':spec['level'] = (0, 1, 3)[level]
            spec['tier'] = level
            directory = run/'fixtures'/f'{family}_{level}'
            directory.mkdir(parents=True, exist_ok=True)
            write(directory/'spec.json', spec)
            existing = directory/'preparation.json'
            record = json.loads(existing.read_text()) if existing.exists() else {}
            if record.get('status') != 'complete':
                if existing.exists():shutil.copy2(existing, directory/'preparation_failed.json')
                record = invoke(run, directory, existing, 'baseline_source', 'prepare', {}, 'sdirk3_mrsav')
            fixtures.append(dict(id=directory.name, spec=spec, preparation=record))
            print(directory.name, record['status'], record.get('velocity_dofs'), flush=True)
            write(run/'preparation-progress.json', fixtures)
            if record['status'] != 'complete':raise RuntimeError(f'Fixture failed: {directory}')
    write(run/'campaign.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        candidate_hashes=hashes, baseline_manifest='baseline.json', fixtures=fixtures,
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            processor=platform.processor(), thread_environment=THREAD_ENV),
        limits=['COMM_SELF, no MPI scaling', 'GAMG only; local PETSc has no Hypre',
                'Short performance workloads; no long-time physical accuracy claim']))


def measure(run: Path, levels: list[int], variants: list[str], workloads: list[str], repeats: int, scheme: str, timeout: float = 120.) -> None:
    campaign = json.loads((run/'campaign.json').read_text())
    jobs = [(repeat, fixture['id'], workload, variant)
        for repeat in range(repeats) for fixture in campaign['fixtures'] if fixture['spec']['tier'] in levels
        for workload in workloads for variant in variants[repeat % len(variants):]+variants[:repeat % len(variants)]]
    for index, (repeat, fixture, workload, variant) in enumerate(jobs):
        output = run/'samples'/fixture/workload/scheme/variant/f'{repeat}.json'
        if output.exists():
            result = json.loads(output.read_text())
            expected_source, expected_options = VARIANTS[variant]
            if result.get('status') == 'complete' and (result.get('options') != expected_options or
                    result.get('source') != str((run/expected_source).resolve()) or
                    result.get('scheme') != scheme):
                raise ValueError('Existing sample has different source/options; use a new campaign')
        elif repeat and any(json.loads(p.read_text()).get('status') == 'timeout' for p in output.parent.glob('*.json')):
            result = dict(status='not_run', error='Earlier repeat exceeded the recorded time limit')
            output.parent.mkdir(parents=True, exist_ok=True)
            write(output, result)
        else:
            source, options = VARIANTS[variant]
            result = invoke(run, run/'fixtures'/fixture, output, source, workload, options, scheme, timeout)
        print(index+1, '/', len(jobs), fixture, workload, variant, repeat, result['status'],
              round(result.get('online_seconds', 0.), 4), flush=True)
        write(run/'measurement-progress.json', dict(completed=index+1, total=len(jobs), last=str(output),
            status='complete' if index+1==len(jobs) else 'running', updated_utc=datetime.now(timezone.utc).isoformat()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'measure'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--levels', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--workloads', nargs='+', choices=('matrix', 'fixed', 'replay', 'adaptive', 'step'), default=['replay'])
    parser.add_argument('--timeout', type=float, default=120.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--scheme', choices=('sdirk2_mrsav', 'sdirk3_mrsav'), default='sdirk3_mrsav')
    args = parser.parse_args()
    if args.timeout <= 0:parser.error('timeout must be positive')
    if args.repeats < 1:parser.error('repeats must be positive')
    run = args.run.resolve()
    if args.action == 'prepare':prepare(run)
    else:measure(run, args.levels, args.variants, args.workloads, args.repeats, args.scheme, args.timeout)


if __name__ == '__main__':
    main()

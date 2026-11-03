"""Compare a captured pre-change Legendre backend with the active implementation.

Same continuum MMS, modes, forcing, schemes and prescribed schedules; interleaved
single-thread repeats. Primary timers exclude error reconstruction, profiling and
storage. Profiles are separate instrumented integrations, not additive cost shares.
The analyzer reads saved modal fields and never runs a solver.
"""
import argparse
import cProfile
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter, process_time
from typing import Any, cast
from uuid import uuid4

import numpy as np

from experiments.legendre_mac_benchmark.run import micro_cpu, schedules
from experiments.problems import _expressions
from experiments.records import PROJECT, digest, provenance, write_json
from solver.integrate import integrate
from solver.model import Model
from solver.schemes.registry import make_scheme, scheme_info
from solver.spectral.assembly import Space
from solver.spectral.model import SpectralModel

CONFIG = Path(__file__).parent/'configs/first_round.json'


def capture_source(destination: Path, source: Path = PROJECT/'solver/spectral') -> None:
    """Capture exact source before editing, or copy a previously captured baseline."""
    destination.mkdir(parents=True)
    files = {}
    for path in sorted(source.glob('*.py')):
        shutil.copy2(path, destination/path.name)
        files[path.name] = digest(destination/path.name)
    write_json(destination/'source.json', {'files': files})


def load_baseline(directory: Path) -> tuple[Any, Any]:
    """Load the captured package under solver, sharing only unchanged core/seam code."""
    metadata = json.loads((directory/'source.json').read_text())
    for name, checksum in metadata['files'].items():
        if digest(directory/name) != checksum:
            raise ValueError(f'Baseline checksum mismatch: {name}')
    name = 'solver._legendre_baseline_'+digest(directory/'source.json')[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory/'__init__.py',
                                                     submodule_search_locations=[str(directory)])
        if spec is None or spec.loader is None:
            raise ValueError('Cannot load baseline package')
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    return (importlib.import_module(name+'.assembly').Space,
            importlib.import_module(name+'.model').SpectralModel)


def build_model(config: dict[str, Any], size: int,
                constructors: tuple[Any, Any]) -> tuple[Model, Any]:
    space_class, model_class = constructors
    lx, ly, nu, amplitude = (config[key] for key in ('lx', 'ly', 'nu', 'amplitude'))
    expressions = _expressions(lx, ly, nu)
    options = {'tolerance': config['stokes_tolerance'], 'cache_size': config['cache_size'],
               'dealias': config['dealias']}
    if model_class is SpectralModel:
        options.update(stokes_backend=config.get('optimized_backend', 'direct'),
                       krylov_tolerance=config.get('krylov_tolerance', 1e-12),
                       max_iterations=config.get('max_iterations', 200))
    model = model_class(space_class(size, lx, ly), nu, **options)
    extra = max(config['projection_extra'], model.quadrature_extra)
    def projected(index: int, scale: float) -> Any:
        return model.project(lambda x, y: scale*expressions[index](x, y, 0.), extra=extra)
    velocity = np.r_[projected(0, amplitude), projected(1, amplitude)]
    linear = np.r_[projected(7, amplitude), projected(9, amplitude)]
    quadratic = np.r_[projected(8, amplitude**2), projected(10, amplitude**2)]
    model.force = lambda t: np.exp(-t)*linear+np.exp(-2*t)*quadratic
    return cast(Model, model), model.state(0., velocity)


def run_case(config: dict[str, Any], size: int, scheme: str, schedule: str,
             constructors: tuple[Any, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cpu, wall = process_time(), perf_counter()
    model, initial = build_model(config, size, constructors)
    setup_cpu, setup_wall = process_time()-cpu, perf_counter()-wall
    steps = schedules(config)[schedule]
    cpu, wall = process_time(), perf_counter()
    result = integrate(model, make_scheme(scheme, config['gamma']), initial, steps)
    integration_cpu, integration_wall = process_time()-cpu, perf_counter()-wall
    if result.status != 'complete' or len(result.times) != len(steps)+1:
        raise RuntimeError(f'{size}/{scheme}/{schedule}: {result.error}')
    backend = cast(Any, model).stokes
    record = {
        'setup_cpu_seconds': setup_cpu, 'setup_wall_seconds': setup_wall,
        'integration_cpu_seconds': integration_cpu, 'integration_wall_seconds': integration_wall,
        'total_cpu_seconds': setup_cpu+integration_cpu,
        'total_wall_seconds': setup_wall+integration_wall,
        'factorizations': backend.factorizations,
        'assembly_and_factorization_wall_seconds': backend.factorization_seconds,
        'max_stage_residual': max(s['residual'] for stages in result.stages for s in stages),
        'max_continuity_residual': max(s['continuity_residual'] for stages in result.stages for s in stages),
        'max_physical_divergence': max(d['divergence_inf'] for d in result.diagnostics),
        'final_time': result.final.t, 'final_r': result.final.r,
    }
    for attribute in ('assembly_seconds', 'lu_factorization_seconds', 'solve_seconds',
                      'check_seconds', 'right_hand_sides', 'solve_calls',
                      'total_iterations', 'schur_rhs', 'helmholtz_applications'):
        if hasattr(backend, attribute):
            record[attribute] = getattr(backend, attribute)
    final = model.vector(result.final)
    count = config['micro_repeats']
    record['nonlinear_cpu_seconds'] = micro_cpu(lambda: model.nonlinear(final), count)
    record['diagnostics_cpu_seconds'] = micro_cpu(lambda: model.diagnostics(result.final), count)
    shift = config['nu']*float(steps[-1])
    record['cached_stokes_cpu_seconds'] = micro_cpu(
        lambda: model.solve(final, mass=1., viscosity=shift), count)
    lu_bytes = matrix_bytes = 0
    for lu, matrix in backend.cache.values():
        if hasattr(lu, 'L'):
            for factor in (lu.L, lu.U):
                lu_bytes += factor.data.nbytes+factor.indices.nbytes+factor.indptr.nbytes
        if matrix is not None:
            matrix_bytes += matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
    record.update(retained_lu_bytes=lu_bytes, retained_stokes_matrix_bytes=matrix_bytes)
    capacity = scheme_info(scheme).n_roots
    roots = np.full((len(result.final_stages), capacity), np.nan)
    for index, stage in enumerate(result.final_stages):
        roots[index, :len(stage.candidates)] = stage.candidates
    fields = {'velocity': final, 'pressure': np.array([s.pressure for s in result.final_stages]),
              'stage_r': np.array([s.r for s in result.final_stages]), 'roots': roots,
              'root_count': np.array([len(s.candidates) for s in result.final_stages]),
              'times': np.array(result.times),
              'diagnostics': np.array([[d[key] for key in sorted(d)] for d in result.diagnostics])}
    cast(Any, model).close()
    return record, fields


def profile_case(config: dict[str, Any], constructors: tuple[Any, Any]) -> list[dict[str, Any]]:
    size = max(config['legendre_modes'])
    model, initial = build_model(config, size, constructors)
    profiler = cProfile.Profile()
    spectral_directory = Path(cast(str, sys.modules[constructors[1].__module__].__file__)).parent
    result = profiler.runcall(integrate, model, make_scheme('sdirk3_mrsav', config['gamma']),
                             initial, schedules(config)['variable'])
    if result.status != 'complete':
        raise RuntimeError(result.error)
    rows = []
    for entry in profiler.getstats():
        code = entry.code
        if isinstance(code, str):
            filename, function, line = '', code, 0
        else:
            filename, function, line = code.co_filename, code.co_name, code.co_firstlineno
        if Path(filename).parent == spectral_directory \
                or filename.endswith('/stokes.py') or 'SuperLU' in function \
                or function in ('splu', 'leggauss'):
            rows.append({'file': filename, 'function': function, 'line': line,
                         'calls': entry.callcount, 'exclusive_wall_seconds': entry.inlinetime,
                         'inclusive_wall_seconds': entry.totaltime})
    cast(Any, model).close()
    return sorted(rows, key=lambda row: row['inclusive_wall_seconds'], reverse=True)


def _memory_worker(output: str, config: dict[str, Any], baseline: str, variant: str) -> None:
    constructors = load_baseline(Path(baseline)) if variant == 'baseline' else (Space, SpectralModel)
    record, _ = run_case(config, max(config['legendre_modes']), 'sdirk3_mrsav', 'variable', constructors)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(Path(output), {'variant': variant, 'size': max(config['legendre_modes']),
                             'scheme': 'sdirk3_mrsav', 'schedule': 'variable',
                             'whole_worker_peak_rss_bytes': peak if sys.platform == 'darwin' else peak*1024,
                             'retained_lu_bytes': record['retained_lu_bytes'],
                             'retained_stokes_matrix_bytes': record['retained_stokes_matrix_bytes']})


def run_benchmark(config: dict[str, Any], baseline: Path, *, root: Path = PROJECT) -> Path:
    """Persist paired sources, all repeats/fields, profiles and isolated memory samples."""
    if min(config['steps'], config['repeats'], config['micro_repeats']) < 1:
        raise ValueError('Positive steps and repeats required')
    for scheme in config['schemes']:
        scheme_info(scheme)
    load_baseline(baseline)  # verify the supplied capture before copying it
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory = root/'runs/legendre_optimization'/identifier
    directory.mkdir(parents=True)
    capture_source(directory/'baseline_source', baseline)
    capture_source(directory/'optimized_source')
    constructors = {'baseline': load_baseline(directory/'baseline_source'),
                    'optimized': (Space, SpectralModel)}
    write_json(directory/'config.json', config)
    write_json(directory/'schedules.json', {k: v.tolist() for k, v in schedules(config).items()})
    manifest: dict[str, Any] = {'id': identifier, 'status': 'running', 'source': provenance(),
        'purpose': 'legendre_optimization_equivalence_and_total_cost',
        'completed_cases': 0, 'artifacts': {},
        'baseline_source': json.loads((directory/'baseline_source/source.json').read_text()),
        'optimized_source': json.loads((directory/'optimized_source/source.json').read_text())}
    write_json(directory/'manifest.json', manifest)
    _expressions(config['lx'], config['ly'], config['nu'])
    records, fields = [], {}
    cases = [(size, scheme, schedule) for size in config['legendre_modes']
             for scheme in config['schemes'] for schedule in ('fixed', 'variable')]
    manifest['planned_cases'] = 2*config['repeats']*len(cases)
    try:
        with (directory/'run.log').open('w') as log:
            for repeat in range(config['repeats']):
                for size, scheme, schedule in (cases if repeat % 2 == 0 else list(reversed(cases))):
                    for variant in (('baseline', 'optimized') if repeat % 2 == 0 else ('optimized', 'baseline')):
                        record, arrays = run_case(config, size, scheme, schedule, constructors[variant])
                        key = f'{variant}_{size}_{scheme}_{schedule}_{repeat}'
                        record.update(variant=variant, size=size, scheme=scheme, schedule=schedule,
                                      repeat=repeat, field_key=key)
                        records.append(record)
                        fields.update({key+'__'+name: value for name, value in arrays.items()})
                        manifest['completed_cases'] = len(records)
                        message = f"{len(records)}/{manifest['planned_cases']} {key}: {record['total_cpu_seconds']:.6f}s"
                        log.write(message+'\n')
                        log.flush()
                        print(message, flush=True)
            write_json(directory/'profiles.json', {k: profile_case(config, v) for k, v in constructors.items()})
            if config.get('measure_memory', True):
                for variant in constructors:
                    worker = mp.get_context('spawn').Process(target=_memory_worker, args=(
                        str(directory/f'memory_{variant}.json'), config,
                        str(directory/'baseline_source'), variant))
                    worker.start()
                    worker.join(60.)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
                        raise RuntimeError('Isolated memory sample timed out')
                    if worker.exitcode:
                        raise RuntimeError('Isolated memory sample failed')
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(directory/'results.json', {'records': records})
        np.savez_compressed(directory/'fields.npz', **fields)
        for path in directory.rglob('*'):
            if path.is_file() and path.name != 'manifest.json' and '__pycache__' not in path.parts:
                manifest['artifacts'][str(path.relative_to(directory))] = digest(path)
        write_json(directory/'manifest.json', manifest)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--baseline', type=Path, required=True, help='captured spectral source directory')
    args = parser.parse_args()
    print('RUN='+str(run_benchmark(json.loads(args.config.read_text()), args.baseline.resolve())))


if __name__ == '__main__':
    main()

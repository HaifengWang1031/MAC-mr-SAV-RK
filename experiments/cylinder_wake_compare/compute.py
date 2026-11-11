"""Launch two isolated, persistent SAV cylinder runs from one immutable source copy.

Purpose: contrast Re100 vortex shedding with a higher-Re *two-dimensional* wake.
Geometry, mesh, inlet, SAV scheme and accuracy controls are held fixed. Reynolds
number changes only viscosity; no turbulence model or artificial forcing is used.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from uuid import uuid4

from experiments.records import PROJECT, digest, provenance, write_json
from experiments.cylinder_ns.config import effective_config

DEFAULTS: dict[str, Any] = {
    'reynolds': [100., 1000.], 'resolution': 1., 'circle_segments': 96,
    'scheme': 'sdirk3_mrsav', 'gamma': 1000., 'T': 12.,
    'display_start': 10., 'snapshot_interval': .01, 'fps': 20,
    'vorticity_limit': 5., 'initial_step': .0002,
    'controller': {'kind': 'PI', 'atol': 1e-8, 'rtol': 2e-4, 'max_step': .002,
                   'min_step': 1e-7, 'min_factor': .5, 'max_factor': 1.5},
    'scalar_atol': 1e-5, 'check_duration': .05, 'check_resolution_factor': 1.25,
    'short_time_gate': .01, 'short_space_gate': .05,
}
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')

WORKER = """import json,sys,traceback
from pathlib import Path
from experiments.cylinder_ns.run import run,load_cylinder
from experiments.workflow import write_json
folder=Path(sys.argv[1]); config=json.loads((folder/'input.json').read_text())
try:
    output=run(config,root=folder)
    cfg,manifest=load_cylinder(output,require_complete=True)
    write_json(folder/'result.json',{'status':'complete','run':str(output),'results_sha256':manifest['results_sha256']})
except Exception as error:
    write_json(folder/'result.json',{'status':'failed','error':f'{type(error).__name__}: {error}'})
    traceback.print_exc()
    raise
"""


def prepare(config: dict[str, Any], *, root: Path = PROJECT) -> Path:
    """Resolve common controls and freeze source before any long computation starts."""
    if set(config)-set(DEFAULTS):
        raise ValueError('Unknown cylinder comparison key')
    cfg = {**DEFAULTS, **config}
    if len(cfg['reynolds']) != 2 or not 0 < cfg['reynolds'][0] < cfg['reynolds'][1] \
            or not 0 <= cfg['display_start'] < cfg['T'] or cfg['snapshot_interval'] <= 0 \
            or cfg['fps'] < 1 or cfg['vorticity_limit'] <= 0 or not cfg['scheme'].endswith('_mrsav'):
        raise ValueError('Invalid cylinder comparison range or SAV method')
    count = round((cfg['T']-cfg['display_start'])/cfg['snapshot_interval'])
    if count < 2 or abs(count*cfg['snapshot_interval']-(cfg['T']-cfg['display_start'])) > 1e-10:
        raise ValueError('Display window must contain equally spaced requested snapshots')
    directory = (root/'runs'/'cylinder_wake_compare'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])).resolve()
    directory.mkdir(parents=True)
    source = directory/'source'
    source.mkdir()
    from run_support.source import legacy_source_files
    for path in legacy_source_files(PROJECT):
        target = source/path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    source_hashes = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
    write_json(directory/'source.json', {'files': source_hashes})
    write_json(directory/'config.json', cfg)
    manifest: dict[str, Any] = {'experiment': 'cylinder_wake_compare', 'status': 'prepared',
        'config_sha256': digest(directory/'config.json'), 'source_manifest_sha256': digest(directory/'source.json'),
        'preparation_source': provenance(), 'cases': [], 'claim': 'two-dimensional unsteady wakes; not 3D turbulence'}
    for index, reynolds in enumerate(cfg['reynolds']):
        folder = directory/f'case_{index}'
        folder.mkdir()
        case = {'resolution': cfg['resolution'], 'circle_segments': cfg['circle_segments'],
            'nu': .1/reynolds, 'inlet_max': 1.5, 'scheme': cfg['scheme'], 'gamma': cfg['gamma'],
            'T': cfg['T'], 'mode': 'adaptive', 'initial_step': cfg['initial_step'],
            'controller': cfg['controller'], 'scalar_atol': cfg['scalar_atol'],
            'stokes_backend': 'recycled', 'log_every': 250,
            'snapshots': [cfg['display_start']+j*cfg['snapshot_interval'] for j in range(count+1)]}
        effective_config(case)
        write_json(folder/'input.json', case)
        manifest['cases'].append({'reynolds': reynolds, 'directory': str(folder),
                                  'config_sha256': digest(folder/'input.json'), 'status': 'prepared'})
    write_json(directory/'manifest.json', manifest)
    return directory


def launch(directory: Path) -> dict[str, Any]:
    """Start independent process sessions with file-owned IO, preserving partial runs."""
    directory = directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['status'] != 'prepared':
        raise ValueError('Launch requires a fresh prepared comparison')
    environment = os.environ.copy()
    environment.update({key: '1' for key in THREADS})
    environment['MPLBACKEND'] = 'Agg'
    for case in manifest['cases']:
        folder = Path(case['directory'])
        command = [sys.executable, '-c', WORKER, str(folder)]
        with (folder/'worker.log').open('w') as log:
            process = subprocess.Popen(command, cwd=directory/'source', env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        case.update(status='running', pid=process.pid, command=command)
    manifest['status'] = 'running'
    write_json(directory/'manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    directory = prepare({} if args.config is None else json.loads(args.config.read_text()), root=args.root)
    if not args.prepare_only:
        launch(directory)
    print(directory)


if __name__ == '__main__':
    main()

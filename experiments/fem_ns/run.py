"""Compute, finalize, verify and reuse independent FEM records."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter, process_time

import h5py

from experiments.records import PROJECT, begin_run, finish_run, provenance, reusable_run, save_result, fail_run
from experiments.fem_backends import solver_provenance
from experiments.controllers import make_controller
from solver.adaptivity import AdaptiveResult, ScalarErrorControl, integrate_adaptive
from solver.integrate import Result, integrate
from solver.schemes.registry import make_scheme, scheme_info
from .config import DEFAULTS, OPTIONS, effective_config
from .records import load_fem
from .problems import build_model
from .representation import save_representation
from .attempts import save_attempts

def run(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    cfg = effective_config(config)
    source = provenance()
    solver_provenance(source, cfg)
    source['packages']['scikit-fem'] = version('scikit-fem')
    source['packages']['pymetis'] = version('pymetis')
    if cfg.get('stokes_factor_backend') == 'pardiso':
        source['packages'].update({name: version(name) for name in ('pypardiso', 'mkl')})
    parent = Path(root)/'runs'/'fem_ns'
    reused = reusable_run(parent, cfg, source, reader=lambda path: load_fem(path, require_complete=True), rerun=rerun)
    if reused is not None:
        return reused
    directory, manifest = begin_run(parent, cfg, source, metadata={'experiment': 'fem_ns', 'rerun': rerun},
                                    started_key='started')
    (directory/'run.log').write_text(f'Started {manifest["started"]}\n')
    cpu, wall = process_time(), perf_counter()
    try:
        model, initial = build_model(cfg)
        setup_cpu, setup_wall = process_time()-cpu, perf_counter()-wall
        cpu = process_time()
        selected = make_scheme(cfg['scheme'], cfg['gamma'])
        result: Result
        if cfg.get('mode') == 'adaptive':
            scalar = None if cfg['scalar_atol'] is None else ScalarErrorControl(cfg['scalar_atol'], cfg['gamma'])
            result = integrate_adaptive(model, selected, initial, cfg['T'], cfg['initial_step'],
                                        controller=make_controller(cfg['controller'], cfg['scheme']),
                                        scalar_control=scalar, snapshots=cfg['snapshots'])
        else:
            result = integrate(model, selected, initial, cfg['actual_steps'], snapshots=cfg['snapshots'])
        metrics = {**model.performance_metrics(), 'setup_cpu_seconds': setup_cpu, 'setup_wall_seconds': setup_wall,
                   'integration_cpu_seconds': process_time()-cpu, 'integration_wall_seconds': result.seconds,
                   'accepted_steps': len(result.times)-1,
                   'rejected_steps': sum(not a['accepted'] for a in result.attempts) if isinstance(result, AdaptiveResult) else 0}
        pending = directory/'results.pending.h5'
        info = scheme_info(cfg['scheme'])
        save_result(pending, result, metrics, n_stages=info.n_stages, n_roots=info.n_roots,
                    pressure_semantics=info.pressure_semantics)
        with h5py.File(pending, 'a') as data:
            save_representation(data, model.velocity_basis, model.pressure_basis)
            if isinstance(result, AdaptiveResult):
                save_attempts(data, result.attempts)
        pending.replace(directory/'results.h5')
        finish_run(directory, manifest, status=result.status, error=result.error, metrics=metrics)
        with (directory/'run.log').open('a') as stream:
            stream.write(f'{result.status}: {result.error}\n{json.dumps(metrics, indent=2)}\n')
    except Exception as error:
        # Setup/storage failures cannot supply an accepted prefix; retain an auditable attempt.
        fail_run(directory, manifest, error)
        with (directory/'run.log').open('a') as stream:
            stream.write(manifest['error']+'\n')
        raise
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    directory = run(json.loads(args.config.read_text()), root=args.root, rerun=args.rerun)
    print(directory)
    _, manifest = load_fem(directory)
    if manifest['status'] != 'complete':
        raise SystemExit(f"Run failed: {manifest['error']}")


if __name__ == '__main__':
    main()

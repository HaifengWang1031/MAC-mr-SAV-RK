"""One fresh-process workload. The controller freezes this file with both sources."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time
from typing import Any
import traceback

ENTRY = perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--fixture', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--options', default='{}')
parser.add_argument('--workload', choices=('prepare', 'matrix', 'fixed', 'replay', 'adaptive', 'step'), required=True)
parser.add_argument('--scheme', default='sdirk3_mrsav')
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_info
from _cases import make_mesh, load_mesh, build_model
from solver.integrate import Result, integrate
from solver.adaptivity import PIController, integrate_adaptive
from solver.schemes.registry import make_scheme
import solver


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(result: dict[str, Any]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.update(process_wall_seconds=perf_counter()-ENTRY,
        process_cpu_seconds=process_time(),
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
        source=str(args.source.resolve()), solver_file=str(Path(solver.__file__).resolve()),
        threadpools=threadpool_info(), fixture_sha256=digest(args.fixture/'mesh.npz'),
        initial_sha256=digest(args.fixture/'initial.npz'), worker_sha256=digest(Path(__file__)))
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')
    temporary.replace(args.output)


try:
    spec = json.loads((args.fixture/'spec.json').read_text())
    if args.workload == 'prepare':
        mesh = make_mesh(spec)
        args.fixture.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.fixture/'mesh.npz', p=mesh.p, t=mesh.t,
                            **{'boundary_'+k: v for k, v in mesh.boundaries.items()})
        if spec['family'] == 'hele':
            # Zero trace/velocity is independently known to satisfy every B row.
            # Fixture preparation needs the original consistent matrix, not its LU.
            from skfem import asm
            from solver.fem import MeshSpace
            from solver.fem.model import _mass
            vb, pb = MeshSpace(mesh).bases()
            fixed = np.concatenate([mesh.boundaries[k] for k in ('walls', 'airfoil')])
            free = vb.complement_dofs(vb.get_dofs(facets=fixed).all())
            velocity = np.zeros(len(free))
            mass_matrix = asm(_mass, vb).tocsc()[free][:, free].tocsc()
            pressure_dofs = int(pb.N)
        else:
            model = build_model(mesh, spec, dict(stokes_backend='direct', cache_size=2))
            velocity = model.solve(model.zero_velocity(), mass=1., viscosity=model.nu*1e-3).velocity
            mass_matrix = model.mass
            pressure_dofs = int(model.pressure_basis.N)
        np.savez_compressed(args.fixture/'initial.npz', velocity=velocity, t=.1, r=.03)
        sp.save_npz(args.fixture/'mass.npz', mass_matrix)
        write(dict(status='complete', velocity_dofs=len(velocity), pressure_dofs=pressure_dofs,
                   triangles=mesh.t.shape[1]))
    else:
        start = perf_counter()
        model = build_model(load_mesh(args.fixture/'mesh.npz'), spec,
                            dict(cache_size=2, **json.loads(args.options)))
        init = perf_counter()-start
        with np.load(args.fixture/'initial.npz') as data:
            initial = model.state(float(data['t']), data['velocity'], float(data['r']))
        result = dict(status='complete', initialization_seconds=init, workload=args.workload,
                      scheme=args.scheme, options=json.loads(args.options),
                      velocity_dofs=len(model.free_dofs), pressure_dofs=model.pressure_basis.N)
        if args.workload == 'matrix':
            coordinates = model.velocity_basis.doflocs[:, model.free_dofs]
            velocity = np.column_stack([.01*np.sin(coordinates[0]+coordinates[1]),
                                        .02*np.cos(coordinates[0]-coordinates[1])])
            xy = model.pressure_basis.doflocs
            pressure = np.column_stack([np.cos(xy[0]+xy[1]), np.sin(xy[0]-2*xy[1])])
            if model.pressure_gauge:
                pressure -= (model.pressure_weights @ pressure)/sum(model.pressure_weights)
            constraint = model.divergence @ velocity
            records = []
            for mass, visc in [(1., model.nu*2.5e-7), (1., model.nu*2.525e-7),
                               (1., model.nu*.0025), (1., model.nu*.002525), (0., model.nu)]:
                load = mass*(model.mass @ velocity)+visc*(model.linear_stiffness @ velocity)-model.divergence.T @ pressure
                weak = json.loads(args.options).get('fem_load_mode') == 'weak'
                rhs = load if weak else model._mass_factor.solve(load)
                before = model.performance_metrics()
                start, cpu = perf_counter(), process_time()
                function = model.stokes.solve_load_columns if weak else model.stokes.solve_columns
                solved = function(list(rhs.T), mass=mass, viscosity=visc, divergence_rhs=constraint)
                elapsed, cpu_elapsed = perf_counter()-start, process_time()-cpu
                actual_v = np.column_stack([s.velocity for s in solved])
                actual_p = np.column_stack([s.pressure for s in solved])
                records.append(dict(mass=mass, viscosity=visc, wall_seconds=elapsed, cpu_seconds=cpu_elapsed,
                    velocity_error_inf=float(np.max(abs(actual_v-velocity))),
                    pressure_error_inf=float(np.max(abs(actual_p-pressure))),
                    metrics={k: v-before.get(k, 0) for k,v in model.performance_metrics().items()}))
            result['matrix_solves'] = records
            result['online_seconds'] = sum(row['wall_seconds'] for row in records)
        else:
            scheme = make_scheme(args.scheme)
            start, cpu = perf_counter(), process_time()
            integration: Result
            if args.workload == 'adaptive':
                integration = integrate_adaptive(model, scheme, initial, .0012, .0001,
                    controller=PIController(atol=1e-8, rtol=1e-5, max_step=.00025,
                                            min_step=1e-10, max_steps=2000))
            else:
                steps = ([.0001] if args.workload=='step' else [.0001]*12 if args.workload == 'fixed' else
                         [.0001*(1.01**j) for j in range(12)])
                integration = integrate(model, scheme, initial, steps)
            attempts = getattr(integration, 'attempts', [])
            nonfinite_attempts = [{
                'index': i, 'key': key, 'value': str(value)}
                for i, row in enumerate(attempts) for key, value in row.items()
                if isinstance(value, (float, np.floating)) and not np.isfinite(value)]
            saved_attempts = [{key: (None if isinstance(value, (float, np.floating)) and
                not np.isfinite(value) else value) for key, value in row.items()} for row in attempts]
            result.update(online_seconds=perf_counter()-start, online_cpu_seconds=process_time()-cpu,
                status=integration.status, error=integration.error, times=integration.times,
                stages=integration.stages, attempts=saved_attempts, nonfinite_attempt_diagnostics=nonfinite_attempts,
                observer_seconds=integration.observer_wall_seconds,
                final_r=integration.final.r)
            result['metrics'] = model.performance_metrics()
            result['phase'] = 'pressure_recovery'
            write({**result, 'status': 'running', 'integration_status': result['status']})
            start = perf_counter()
            if hasattr(model, 'instantaneous_solution'):
                physical_pressure = model.instantaneous_solution(model.vector(integration.final), integration.final.t)[1]
            else:
                physical_pressure = np.empty(0)
            result['pressure_recovery_seconds'] = perf_counter()-start
            result['phase'] = 'complete'
            np.savez_compressed(args.output.with_suffix('.npz'), velocity=model.vector(integration.final),
                stage_pressure=np.array([s.pressure for s in integration.final_stages]),
                stage_r=np.array([s.r for s in integration.final_stages]), physical_pressure=physical_pressure)
        result['metrics'] = model.performance_metrics()
        write(result)
except Exception as error:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='failed', error=f'{type(error).__name__}: {error}',
        traceback=traceback.format_exc(), process_wall_seconds=perf_counter()-ENTRY), indent=2)+'\n')
    raise

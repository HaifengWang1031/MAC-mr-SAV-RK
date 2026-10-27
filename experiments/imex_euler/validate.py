"""Fixed-space temporal refinement across six backends; saved fields, read-only analysis.

Only h varies within each group. Fourier/channel use existing independent MMS loads;
MAC/Legendre use a known semidiscrete trajectory with loads from fixed operators.
This verifies temporal implementation, not new spatial accuracy or long-time stability.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from experiments.workflow import PROJECT, digest, provenance, write_json
from experiments.manufactured_convergence.model import spatial_velocity
from experiments.channel_ns.problems import DiscreteManufactured
from experiments.periodic_ns.problems import Manufactured
from solver.channel import ChannelGrid, ChannelModel
from solver.fourier import FourierModel, FourierSpace
from solver.integrate import integrate
from solver.mac.grid import MACGrid
from solver.mac.tensor_stokes import TensorStokes
from solver.mac_ns import MACNavierStokes
from solver.schemes.registry import make_scheme
from solver.spectral.assembly import Space
from solver.spectral.model import SpectralModel

BACKENDS = ('mac_direct', 'mac_tensor', 'legendre', 'fourier', 'channel_direct', 'channel_fft')
NAMES = ('imex_euler', 'imex_euler_mrsav')
DEFAULTS: dict[str, Any] = {'backends': list(BACKENDS), 'schemes': list(NAMES),
                          'counts': [80, 160, 320], 'gammas': [1., 1000.],
                          'nx': 12, 'ny': 10, 'legendre_size': 6, 'T': .4, 'amplitude': .1}


def _setup(backend: str, cfg: dict) -> tuple[Any, Any, np.ndarray, str, str]:
    model: Any
    if backend.startswith('channel'):
        model = ChannelModel(ChannelGrid(cfg['nx'], cfg['ny']), .1,
                             backend='fft' if backend == 'channel_fft' else 'direct')
        ref = DiscreteManufactured(model.grid, .1, cfg['amplitude'])
        model.force = ref.force
        return model, ref.exact, np.full(ref.exact(0.).shape, model.grid.area), 'diagonal', 'independent_numpy_discrete_mms'
    if backend == 'fourier':
        model = FourierModel(FourierSpace(24, 18), .1)
        ref_fourier = Manufactured(model.space, .1, cfg['amplitude'])
        model.force = lambda t: model.space.to_spectral(ref_fourier.force(t))
        exact = lambda t: model.space.to_spectral(ref_fourier.exact(t))
        norm = np.broadcast_to(model.space.area*model.space.weights, exact(0.).shape)
        return model, exact, norm, 'diagonal', 'independent_continuous_resolved_fourier_mms'
    if backend == 'legendre':
        model = SpectralModel(Space(cfg['legendre_size']), .1)
        # Polynomial streamfunction 16 x²(1-x)² y²(1-y)²; exact wall compatibility.
        base = np.concatenate([
            model.project(lambda x, y: 32*x*x*(1-x)**2*y*(1-y)*(1-2*y)),
            model.project(lambda x, y: -32*x*(1-x)*(1-2*x)*y*y*(1-y)**2)])
        norm, kind = model.mass.toarray(), 'component_mass'
    else:
        model = MACNavierStokes(MACGrid(cfg['nx'], cfg['ny']), .1)
        if backend == 'mac_tensor': model.backend = TensorStokes(model.ops)
        base = spatial_velocity(model.grid)
        norm, kind = np.full(base.shape, model.grid.area), 'diagonal'
    stiffness, convection = model.apply_K(base), model.nonlinear(base)
    def amplitude(t: float) -> float:
        return float(cfg['amplitude']*np.exp(-t)*(1+.25*np.sin(t)))
    def derivative(t: float) -> float:
        return float(cfg['amplitude']*np.exp(-t)*(-1-.25*np.sin(t)+.25*np.cos(t)))
    model.force = lambda t: derivative(t)*base+model.nu*amplitude(t)*stiffness+amplitude(t)**2*convection
    return model, lambda t: amplitude(t)*base, norm, kind, 'semidiscrete_mms_from_fixed_model_operators'


def _error(delta: np.ndarray, norm: np.ndarray, kind: str) -> float:
    if kind == 'component_mass':
        x, y = np.split(delta, 2)
        squared = float(x@norm@x+y@norm@y)
    else:
        squared = float(np.sum(norm*np.abs(delta)**2))
    return float(np.sqrt(max(squared, 0.)))


def validate(config: dict | None = None, *, root: Path = PROJECT) -> Path:
    supplied = config or {}
    if set(supplied)-set(DEFAULTS): raise ValueError('Unknown IMEX validation configuration')
    cfg = {**DEFAULTS, **supplied}
    for key, choices in (('backends', BACKENDS), ('schemes', NAMES)):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]) or any(n not in choices for n in cfg[key]):
            raise ValueError(f'Invalid {key}')
    if len(cfg['counts']) < 2 or any(type(n) is not int or n < 1 for n in cfg['counts']) \
            or any(b <= a for a, b in zip(cfg['counts'], cfg['counts'][1:])):
        raise ValueError('Counts must be increasing positive integers')
    if not cfg['gammas'] or len(set(cfg['gammas'])) != len(cfg['gammas']) \
            or any(not np.isfinite(g) or g < 0 for g in cfg['gammas']) \
            or not np.isfinite([cfg['T'], cfg['amplitude']]).all() or min(cfg['T'], cfg['amplitude']) <= 0:
        raise ValueError('Invalid gamma/time/amplitude')
    MACGrid(cfg['nx'], cfg['ny']); Space(cfg['legendre_size'])
    directory = root/'runs'/'imex_euler_validation'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    directory.mkdir(parents=True)
    write_json(directory/'config.json', cfg)
    manifest = {'experiment': 'imex_euler_validation', 'status': 'running', 'source': provenance(),
                'config_sha256': digest(directory/'config.json')}
    write_json(directory/'manifest.json', manifest)
    rows: list[dict] = []
    try:
        with (directory/'run.log').open('w', buffering=1) as log:
            for backend in cfg['backends']:
                model, exact, norm, norm_kind, reference = _setup(backend, cfg)
                for name in cfg['schemes']:
                    for gamma in (cfg['gammas'] if name.endswith('_mrsav') else [0.]):
                        members = []
                        for count in cfg['counts']:
                            result = integrate(model, make_scheme(name, gamma), model.state(0., exact(0.)),
                                               [cfg['T']/count]*count)
                            if result.status != 'complete': raise RuntimeError(result.error)
                            artifact = f'{backend}_{name}_g{gamma:g}_n{count}.npz'
                            stages = [s for step in result.stages for s in step]
                            np.savez_compressed(directory/artifact, velocity=model.vector(result.final),
                                                exact=exact(result.final.t), norm=norm,
                                                u=result.final.u, v=result.final.v, r=result.final.r,
                                                times=result.times, pressure=result.final_stages[0].pressure,
                                                residual=[s['residual'] for s in stages],
                                                scalar_residual=[s['scalar_residual'] for s in stages],
                                                continuity=[s['continuity_residual'] for s in stages],
                                                physical_divergence=[s['divergence_inf'] for s in stages])
                            members.append({'count': count, 'artifact': artifact,
                                            'sha256': digest(directory/artifact)})
                        rows.append({'backend': backend, 'scheme': name, 'gamma': gamma,
                                     'norm_kind': norm_kind, 'reference': reference, 'members': members})
                        log.write(f'complete {backend} {name} gamma={gamma:g}\n')
                print(f'IMEX validation complete: {backend}', flush=True)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(directory/'results.json', {'rows': rows})
        manifest['results_sha256'] = digest(directory/'results.json')
        write_json(directory/'manifest.json', manifest)
        with (directory/'run.log').open('a') as log:
            log.write(f"{manifest['status']}: {manifest.get('error', '')}\n")
    return directory


def analyze_validation(directory: Path, *, root: Path = PROJECT) -> Path:
    """Recompute norms and rates from explicit saved fields, without model construction."""
    directory = directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('experiment') != 'imex_euler_validation' or manifest['status'] != 'complete':
        raise ValueError('Complete IMEX validation required')
    if digest(directory/'config.json') != manifest['config_sha256'] \
            or digest(directory/'results.json') != manifest['results_sha256']:
        raise ValueError('IMEX validation checksum mismatch')
    cfg = json.loads((directory/'config.json').read_text())
    rows = json.loads((directory/'results.json').read_text())['rows']
    expected = {(b, s, g) for b in cfg['backends'] for s in cfg['schemes']
                for g in (cfg['gammas'] if s.endswith('_mrsav') else [0.])}
    if len(rows) != len(expected) or {(r['backend'], r['scheme'], r['gamma']) for r in rows} != expected:
        raise ValueError('Missing/duplicate IMEX validation groups')
    summary = []
    for row in rows:
        errors: list[float] = []
        scalars: list[float] = []
        residuals: list[float] = []
        scalar_residuals: list[float] = []
        divergences: list[float] = []
        physical: list[float] = []
        if [m['count'] for m in row['members']] != cfg['counts']: raise ValueError('IMEX member counts mismatch')
        for member in row['members']:
            path = directory/member['artifact']
            if digest(path) != member['sha256']: raise ValueError('IMEX member checksum mismatch')
            with np.load(path) as data:
                if len(data['times']) != member['count']+1 or not np.isclose(data['times'][-1], cfg['T']):
                    raise ValueError('Invalid IMEX member endpoint')
                errors.append(_error(data['velocity']-data['exact'], data['norm'], row['norm_kind']))
                scalars.append(abs(float(data['r'])))
                for key, values in (('residual', residuals), ('scalar_residual', scalar_residuals),
                                    ('continuity', divergences), ('physical_divergence', physical)):
                    values.append(float(np.max(data[key])))
        counts = np.array(cfg['counts'])
        rates = np.log(np.array(errors[:-1])/errors[1:])/np.log(counts[1:]/counts[:-1])
        summary.append({**row, 'velocity_L2_errors': errors, 'velocity_rates': rates.tolist(),
                        'scalar_absolute_errors': scalars, 'max_momentum_residual': max(residuals),
                        'max_scalar_residual': max(scalar_residuals), 'max_continuity_residual': max(divergences),
                        'max_physical_divergence': max(physical)})
    report = root/'reports'/'imex_euler_validation'/directory.name
    report.mkdir(parents=True, exist_ok=True)
    write_json(report/'analysis.json', {'run': str(directory), 'analysis_source': provenance(),
                                       'manifest_sha256': digest(directory/'manifest.json'), 'rows': summary,
                                       'first_order_observed': all(min(r['velocity_rates']) > .9 for r in summary),
                                       'limits': ['fixed-space temporal tests, not a new spatial convergence study',
                                                  'scalar errors are separate from velocity errors',
                                                  'no long-time stability or physical pressure-order claim']})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'analyze'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    if args.mode == 'run':
        print(validate(None if args.config is None else json.loads(args.config.read_text()), root=args.root))
    else:
        if args.run is None: parser.error('analyze requires --run')
        print(analyze_validation(args.run, root=args.root))


if __name__ == '__main__': main()

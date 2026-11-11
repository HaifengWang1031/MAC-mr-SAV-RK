"""Compute forced MHD: vary mu at fixed Re/box; save physical fluxes and balances.

Purpose: reproduce cascade direction changes and compare instantaneous transport
with long-window statistics, not infer critical exponents from a coarse pilot.
"""
import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import h5py
import numpy as np

from experiments.records import (
    PROJECT, begin_run, fail_run, finish_run, provenance, reusable_run, save_result, write_json,
)
from solver.core import State
from solver.fourier import FourierSpace, MHDState
from solver.fourier.forced_mhd import (ForcedFourierMHD, LOWPASS_SPECTRA, SAMPLED_SPECTRA,
                                      transfer_spectra)
from solver.integrate import integrate, step_sizes
from solver.schemes.registry import make_scheme, scheme_info
from .config import effective_config
from .problem import initial_state, paper_force
from .records import load


def _mhd(state: State) -> MHDState:
    if not isinstance(state, MHDState):
        raise TypeError('Expected MHD state')
    return state


def run(config: dict, *, root: Path = PROJECT, rerun: bool = False) -> Path:
    c = effective_config(config)
    source = provenance()
    parent = root/'runs'/'inverse_cascade'
    reused = reusable_run(parent, c, source, reader=load, rerun=rerun)
    if reused is not None:
        return reused
    directory, manifest = begin_run(parent, c, source, metadata={'experiment': 'inverse_cascade'})
    log = directory/'run.log'
    log.write_text(f'Started {manifest["started"]}\n{json.dumps(c, indent=2)}\n')
    try:
        s = FourierSpace(c['nx'], c['ny'])
        forcing = paper_force(s, c['kf'], c['f0'], c['mu'])
        model = ForcedFourierMHD(s, c['nu_plus'], c['nu_minus'], force=lambda t: forcing)
        initial = initial_state(model, c['kf'], c['f0'], c['mu'], c['seed'], c['noise_rms'])
        samples: list[dict[str, Any]] = []
        count = 0

        def sample(state: State) -> None:
            nonlocal count
            if count % c['sample_every'] == 0:
                samples.append({'time': state.t,
                                **transfer_spectra(model, model.vector(_mhd(state)), state.t)})
            count += 1

        def progress(n: int, t: float, elapsed: float) -> None:
            if n % c['log_every'] == 0:
                with log.open('a') as stream:
                    stream.write(f'accepted={n} t/t0={t/c["t0"]:.9g} integration_wall={elapsed:.3f}s\n')

        sample(initial)
        terminal = c['T']*c['t0']
        result = integrate(model, make_scheme(c['scheme'], c['gamma']/c['t0']), initial,
                           step_sizes(terminal, dt=c['dt']*c['t0']), snapshots=[0., terminal],
                           progress=progress, on_accept=sample)
        metrics = {'accepted_steps': len(result.times)-1, 'integration_wall_seconds': result.seconds,
                   'observer_wall_seconds': result.observer_wall_seconds}
        tolerance = c.get('coupling_tolerance')
        if tolerance is not None and result.status == 'complete':
            deviation = float(np.max(np.abs([stage['frozen_factor']*(1-stage['r'])-1
                                             for row in result.stages for stage in row]))) \
                if result.stages else 0.
            metrics['max_abs_coupling_minus_one'] = deviation
            if not np.isfinite(deviation) or deviation > tolerance:
                raise RuntimeError(f'SAV coupling deviation {deviation:.3e} exceeds '
                                   f'coupling_tolerance {tolerance:.3e}')
        pending = directory/'results.pending.h5'
        info = scheme_info(c['scheme'])
        save_result(pending, result, metrics, n_stages=info.n_stages, n_roots=info.n_roots,
                    pressure_semantics=c['pressure_semantics'], include_stage_pressure=False)
        if not isinstance(result.final, MHDState):
            raise TypeError('Expected MHD final state')
        with h5py.File(pending, 'a') as d:
            d.create_dataset('final/magnetic', data=result.final.magnetic)
            d.create_dataset('snapshots/magnetic', data=np.array([_mhd(v).magnetic for v in result.snapshots]),
                             compression='gzip')
            d.create_dataset('spectra/K', data=samples[0]['K'])
            d.create_dataset('spectra/times', data=[v['time'] for v in samples])
            for key in SAMPLED_SPECTRA+LOWPASS_SPECTRA:
                d.create_dataset('spectra/'+key, data=np.array([v[key] for v in samples]), compression='gzip')
        pending.replace(directory/'results.h5')
        finish_run(directory, manifest, status=result.status, error=result.error, metrics=metrics)
        with log.open('a') as stream:
            stream.write(f'{result.status}: {result.error}\n{json.dumps(metrics)}\n')
    except Exception as error:
        fail_run(directory, manifest, error)
        with log.open('a') as stream:
            stream.write(manifest['error']+'\n')
        raise
    return directory


def run_scan(config: dict, *, root: Path = PROJECT) -> Path:
    """Explicit sequential members; preserve failures and reuse only validated exact matches."""
    directory = root/'runs'/'inverse_cascade_scans'/uuid4().hex[:12]
    directory.mkdir(parents=True)
    write_json(directory/'config.json', config)
    batch: dict[str, Any] = {'status': 'running', 'members': []}
    path = directory/'batch.json'
    write_json(path, batch)
    print(path, flush=True)
    try:
        for overrides in config['members']:
            settings = {**config['base'], **overrides}
            member = run(settings, root=root)
            batch['members'].append({'run': str(member.resolve())})
            write_json(path, batch)
            _, manifest = load(member, require_complete=True)
            print(f'mu={settings.get("mu")} {member.name} {manifest["metrics"]}', flush=True)
        batch['status'] = 'complete'
    except Exception as error:
        batch.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(path, batch)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--root', type=Path, default=PROJECT)
    p.add_argument('--scan', action='store_true')
    p.add_argument('--rerun', action='store_true')
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if a.scan:
        if a.rerun:
            p.error('--rerun applies to single runs')
        print(run_scan(config, root=a.root))
    else:
        path = run(config, root=a.root, rerun=a.rerun)
        print(path)
        load(path, require_complete=True)


if __name__ == '__main__':
    main()

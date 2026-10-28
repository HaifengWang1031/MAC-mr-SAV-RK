"""Warm-factor accounting and read-only integrity gate across the four model seams."""
import json

import numpy as np
import pytest

from experiments.step_efficiency.analyze import analyze
from experiments.step_efficiency.run import run_benchmark


def test_warm_step_benchmark_preserves_costs_and_analyzes_saved_fields(tmp_path, monkeypatch):
    directory = run_benchmark({'N': 16, 'schemes': ['imex_euler_mrsav'],
                               'steps': 2, 'repeats': 1, 'warmup_steps': 1}, root=tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('analysis launched a solver')
    monkeypatch.setattr('experiments.step_efficiency.run.build_model', forbidden)
    report = analyze(directory, root=tmp_path)
    result = json.loads((report/'analysis.json').read_text())
    assert len(result['timings']) == 14 and len(result['pairs']) == 6
    for row in result['measurements']:
        assert row['full_cpu_ms'] > row['step_cpu_ms'] > 0
        if row['backend'] in ('mac_direct', 'legendre_direct', 'channel_direct', 'channel_fft'):
            assert row['counters_before']['factorizations'] == 1
            assert row['counters']['factorizations'] == (0 if row['schedule'] == 'fixed' else 2)
    # Independent closed-form Fourier velocity at t=0 is what the warm run restarts from.
    periodic = next(r for r in result['measurements'] if r['backend'] == 'fourier_leray'
                    and r['schedule'] == 'fixed')
    cfg = result['config']
    t = periodic['final_time']
    x = 2*np.pi*np.arange(16)[None, :]/16
    y = 2*np.pi*np.arange(16)[:, None]/16
    a = cfg['amplitude']*np.exp(-t)*(1+.25*np.sin(t))
    exact = a*(2*np.pi/cfg['ly'])*(np.sin(x)+.2*np.sin(2*x))*np.cos(y)
    with np.load(directory/periodic['fields']) as fields:
        assert np.linalg.norm(fields['u']-exact)/np.linalg.norm(exact) < 1e-3
    (directory/periodic['fields']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        analyze(directory, root=tmp_path)

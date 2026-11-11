"""Integrity and representation gate for forced-cascade runs, without a dynamics import."""
from pathlib import Path

import h5py
import numpy as np

from experiments.records import load_record
from solver.fourier.forced_mhd import LOWPASS_SPECTRA, SAMPLED_SPECTRA
from solver.fourier.space import FourierSpace
from solver.schemes.registry import scheme_info
from .config import CASE


def load(directory: Path, *, require_complete: bool = False) -> tuple[dict, dict]:
    c, manifest = load_record(directory, require_complete=require_complete)
    if c.get('case') != CASE or manifest.get('experiment') != 'inverse_cascade':
        raise ValueError('Expected an inverse_cascade record')
    s = FourierSpace(c['nx'], c['ny'])
    if any(c.get(k) != v for k, v in s.metadata().items()):
        raise ValueError('Cascade Fourier representation mismatch')
    with h5py.File(directory/'results.h5', 'r') as d:
        if d.attrs['status'] != manifest['status'] or d.attrs['pressure_semantics'] != 'leray_eliminated':
            raise ValueError('Cascade status/pressure mismatch')
        times, sample_times, K = d['times'][:], d['spectra/times'][:], d['spectra/K'][:]
        if times.ndim != 1 or len(times) < 1 or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError('Invalid accepted times')
        if len(sample_times) < 1 or sample_times[0] != 0 or np.any(np.diff(sample_times) <= 0) \
                or sample_times[-1] > times[-1]+1e-10:
            raise ValueError('Invalid spectral sample times')
        expected_samples = times[::c['sample_every']]
        if sample_times.shape != expected_samples.shape \
                or not np.allclose(sample_times, expected_samples, rtol=0, atol=1e-10):
            raise ValueError('Spectral sampling does not match accepted steps')
        expected_K = np.arange(int(np.ceil(np.sqrt(s.k2[s.mask])).max())+1)
        if not np.array_equal(K, expected_K):
            raise ValueError('Invalid radial cutoffs')
        for key in SAMPLED_SPECTRA:
            if d['spectra/'+key].shape != (len(sample_times), len(K)) \
                    or not np.isfinite(d['spectra/'+key][:]).all():
                raise ValueError('Invalid sampled spectra')
        # Low-pass budget columns are optional so earlier records stay readable.
        present = [key for key in LOWPASS_SPECTRA if 'spectra/'+key in d]
        if present and len(present) != len(LOWPASS_SPECTRA):
            raise ValueError('Incomplete low-pass budget columns')
        for key in present:
            if d['spectra/'+key].shape != (len(sample_times), len(K)) \
                    or not np.isfinite(d['spectra/'+key][:]).all():
                raise ValueError('Invalid low-pass budget columns')
        if d['final/magnetic'].shape != (2, *s.shape) or d['final/u'].shape != s.shape \
                or d['final/v'].shape != s.shape:
            raise ValueError('Invalid final MHD shapes')
        info = scheme_info(c['scheme'])
        counts = d['corrections/count'][:]
        if counts.shape != (len(times)-1, info.n_stages) or np.any(counts != info.n_corrections):
            raise ValueError('Invalid cascade correction counts')
        if info.n_corrections and len(counts):
            values = d['corrections/values'][:]
            if not np.isfinite(values).all() or not np.array_equal(values[..., -1], d['stages/r'][:]) \
                    or not np.allclose(d['corrections/frozen_factor'][:], 1+values[..., 0]+values[..., 0]**2):
                raise ValueError('Invalid cascade fixed scalar corrections')
        if manifest['status'] == 'complete' and not np.isclose(times[-1], c['T']*c['t0'], rtol=0, atol=1e-9):
            raise ValueError('Incomplete cascade time coverage')
    return c, manifest

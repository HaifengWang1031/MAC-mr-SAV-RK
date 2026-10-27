"""Read saved explicit run IDs; compute spectra without constructing a solver."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import find_peaks, periodogram
from .run import write_json


def analyze(run: Path, output: Path, discard: float) -> dict:
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='complete':
        raise ValueError('Only complete runs can be analyzed here')
    with (run/'results.h5').open('rb') as stream:
        checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    if checksum!=manifest['results_sha256']:
        raise ValueError('Saved result checksum mismatch')
    with h5py.File(run/'results.h5') as h:
        if h.attrs['status']!='complete':
            raise ValueError('Result not finalized')
        data=h['diagnostics'][:]
        columns=json.loads(h['diagnostics'].attrs['columns'])
    signal=data[data[:,0]>=discard]
    if len(signal)<16:
        raise ValueError('Insufficient post-transient samples')
    spacing=np.diff(signal[:,0])
    if not np.allclose(spacing,spacing[0],rtol=1e-8,atol=1e-10):
        raise ValueError('Spectrum requires uniform sampling')
    omega=signal[:,columns.index('probe_omega')]
    frequency,power=periodogram(omega,fs=1/spacing[0],window='hann',detrend='constant')
    peaks,_=find_peaks(power)
    order=peaks[np.argsort(power[peaks])[-12:][::-1]]
    result=dict(status='descriptive_only_source_and_refinement_unverified',run=str(run.resolve()),
                checksum=checksum,discard_before=discard,config=manifest['config'],
                samples=len(omega),probe_mean=float(omega.mean()),probe_std=float(omega.std()),
                peaks=[dict(frequency=float(frequency[k]),power=float(power[k])) for k in order])
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'spectrum.npz',frequency=frequency,power=power)
    write_json(output/'analysis.json',result)
    return result


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--discard',type=float,default=500.)
    a=p.parse_args()
    print(json.dumps(analyze(a.run,a.output,a.discard),indent=2))

if __name__=='__main__':
    main()

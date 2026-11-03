"""Explicit bounded startup campaign; common controls, independent saved run IDs.

Purpose: verify implementation on low reduced-inertia foil flow, separating fixed
step refinement, a two-grid sensitivity check, and +/- attack angle comparison.
These are short startup experiments, not a 3D model validation or a proven
grid-independent/steady aerodynamic benchmark. Logs are program-owned.
"""
import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import h5py
import numpy as np

from experiments.records import PROJECT, digest, write_json
from .analyze import analyze
from .records import load
from .run import run


def final_field(directory: Path) -> tuple[Any, Any, np.ndarray]:
    _, _, _, vb, _ = load(directory)
    full = np.zeros(vb.N)
    with h5py.File(directory/'results.h5', 'r') as data:
        for key, indices in zip(('u', 'v'), vb.split_indices()):
            full[indices] = data['final/'+key][:]
    return vb, vb.split_bases()[0], full


def compare(coarse: Path, reference: Path) -> dict[str, Any]:
    """Full reference quadrature; exact P2 coarse values in bounded probe blocks."""
    ca = json.loads((coarse/'config.json').read_text())
    cb = json.loads((reference/'config.json').read_text())
    controls = ('formulation', 'closure_coefficients', 'chord', 'angle', 'gap_ratio',
                'nu', 'rho', 'pressure_drop', 'ramp_time', 'convection', 'outlet_treatment',
                'bounds', 'surface_points', 'T', 'quadrature_order')
    if any(ca[k] != cb[k] for k in controls):
        raise ValueError('Comparison requires matching physical/geometry/time controls')
    ma = json.loads((coarse/'manifest.json').read_text())
    mb = json.loads((reference/'manifest.json').read_text())
    if ma['source'] != mb['source']:
        raise ValueError('Controlled comparison requires matching source/environment')
    vb, scalar, full = final_field(coarse)
    fb, _, fine = final_field(reference)
    value = np.asarray(fb.interpolate(fine)).reshape(2, -1)
    weight = fb.dx.reshape(-1)
    if np.array_equal(vb.mesh.p, fb.mesh.p) and np.array_equal(vb.mesh.t, fb.mesh.t):
        other = np.asarray(vb.interpolate(full)).reshape(2, -1)
    else:
        points = np.asarray(fb.global_coordinates()).reshape(2, -1)
        other = np.empty(value.shape, dtype=float)  # no DiscreteField slice buffer
        cu, cv = (full[indices] for indices in vb.split_indices())
        for start in range(0, points.shape[1], 128):
            stop = min(start+128, points.shape[1])
            probe = scalar.probes(points[:, start:stop])
            other[:, start:stop] = np.stack([probe @ cu, probe @ cv])
    relative = float(np.sqrt(np.sum((other-value)**2*weight)/max(np.sum(value*value*weight), 1e-30)))
    return dict(run=coarse.name, reference=reference.name, velocity_relative_L2=relative,
                quadrature_points=len(weight), limits='Numerical reference quadrature, not an exact error bound')


def campaign(*, root: Path = PROJECT, quick: bool = False) -> Path:
    campaign_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    parent = Path(root)/'runs/hele_shaw_airfoil_campaign'/campaign_id
    report = Path(root)/'reports/hele_shaw_airfoil'/campaign_id
    parent.mkdir(parents=True)
    report.mkdir(parents=True)
    base = json.loads((Path(__file__).parent/'configs/startup.json').read_text())
    base['layer_cells'] = 2
    if quick:
        base.update(gap_ratio=.1, pressure_drop=108., layer_cells=1, surface_points=64,
                    T=.01, ramp_time=.005, snapshots=[0., .005, .01], dt=.001)
    base_dt = base['dt']
    cases = {
        'sav_dt2': {**base, 'dt': 2*base_dt},
        'sav_base': base,
        'ordinary_reference': {**base, 'scheme': 'sdirk2', 'dt': base_dt/2},
        'sav_fine': {**base, 'layer_cells': 2*base['layer_cells']},
        'sav_zero_angle': {**base, 'angle': 0.},
        'sav_negative_angle': {**base, 'angle': -10.},
    }
    records: dict[str, Any] = {}
    write_json(parent/'request.json', dict(cases=cases, quick=quick, report=str(report), purpose=__doc__))
    with (parent/'campaign.log').open('w', buffering=1) as log:
        for label, config in cases.items():
            log.write(f'START {label}\n')
            print('START', label, flush=True)
            directory = run(config, root=root)
            _, manifest, *_ = load(directory, require_complete=False)
            records[label] = dict(run=str(directory), run_id=directory.name, status=manifest['status'],
                                  results_sha256=manifest['results_sha256'], metrics=manifest['metrics'])
            write_json(parent/'state.json', dict(status='running', cases=records))
            log.write(f'DONE {label} {manifest["status"]} {directory.name}\n')
            if manifest['status'] != 'complete':
                raise RuntimeError(f'{label} failed; accepted prefix preserved at {directory}')
            gc.collect()
        pairs = {}
        for label in ['sav_dt2', 'sav_base']:
            print('COMPARE_TIME', label, flush=True)
            pairs[label] = compare(Path(records[label]['run']), Path(records['ordinary_reference']['run']))
            gc.collect()
        print('COMPARE_MESH', flush=True)
        pairs['mesh'] = compare(Path(records['sav_base']['run']), Path(records['sav_fine']['run']))
        summaries = {}
        for label in ['sav_fine', 'sav_zero_angle', 'sav_negative_angle']:
            print('RENDER', label, flush=True)
            summaries[label] = analyze(Path(records[label]['run']), output=report/label, movie=label == 'sav_fine')
            gc.collect()
        result = dict(status='complete', campaign_id=campaign_id, cases=records, comparisons=pairs,
                      analysis=summaries, quick=quick, reference_status='finer_reference_unverified',
                      limits='Short startup only; two spatial grids, two SAV steps and one ordinary reference. '
                             'No finer ordinary reference, 3D validity or steady/grid-independent claim.')
        write_json(report/'campaign.json', result)
        write_json(parent/'state.json', dict(status='complete', report=str(report), cases=records))
        lines = ['# Hele–Shaw翼型启动验证', '',
                 'MWR深度平均惯性Brinkman，闭尾缘NACA0012，压差驱动。',
                 '这是短程启动实验；普通格式参考尚无更细参考确认，两级网格仅作敏感性检查。', '',
                 '|组别|原始运行|三角形|接受步|', '|---|---|---:|---:|']
        for label, record in records.items():
            lines.append(f"|{label}|{record['run_id']}|{record['metrics']['triangles']:g}|{record['metrics']['accepted_steps']}|")
        lines.extend(['', '|对比|速度相对L2差|', '|---|---:|'])
        for label, pair in pairs.items():
            lines.append(f"|{label}|{pair['velocity_relative_L2']:.7g}|")
        lines.extend(['', '[正攻角细网格动画](sav_fine/evolution.mp4)', '',
                      '[正攻角末态](sav_fine/final.png) / [零攻角末态](sav_zero_angle/final.png) / '
                      '[负攻角末态](sav_negative_angle/final.png)'])
        (report/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        log.write(f'COMPLETE {report}\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    print(campaign(root=args.root, quick=args.quick))


if __name__ == '__main__':
    main()

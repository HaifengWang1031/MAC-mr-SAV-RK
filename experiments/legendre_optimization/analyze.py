"""Read paired saved modal fields; reconstruct independently with NumPy Legendre."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

from experiments.workflow import PROJECT, digest, write_json


@lru_cache(maxsize=12)
def _quadrature(size: int, count: int, lx: float, ly: float) -> tuple[Any, ...]:
    z, w = leggauss(count)
    values = legvander(z, size+1).T
    phi = values[:size]-values[2:size+2]
    return phi, values[:size], .25*lx*ly*np.outer(w, w), .5*lx*(z+1), .5*ly*(z+1)


def _physical(coefficients: Any, basis: Any, size: int) -> Any:
    return basis.T @ np.asarray(coefficients).reshape(size, size).T @ basis


def _norm(values: Any, weight: Any) -> float:
    return float(np.sqrt(np.sum(values**2*weight)))


def _error(config: dict[str, Any], size: int, coefficients: Any, time: float, count: int) -> float:
    phi, _, weight, x, y = _quadrature(size, count, config['lx'], config['ly'])
    xx, yy = np.meshgrid(x, y)
    sx, sy = np.sin(np.pi*xx/config['lx']), np.sin(np.pi*yy/config['ly'])
    cx, cy = np.cos(np.pi*xx/config['lx']), np.cos(np.pi*yy/config['ly'])
    amplitude = config['amplitude']*np.exp(-time)
    exact = np.array([amplitude*2*np.pi/config['ly']*sx**2*sy*cy,
                      -amplitude*2*np.pi/config['lx']*sx*cx*sy**2])
    values = np.array([_physical(c, phi, size) for c in coefficients.reshape(2, -1)])
    return _norm(values-exact, weight)/_norm(exact, weight)


def analyze(directory: Path, *, root: Path = PROJECT) -> Path:
    """Validate completeness/checksums, compare every repeat and report CPU medians."""
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Optimization benchmark is incomplete')
    for name, checksum in manifest['artifacts'].items():
        if digest(directory/name) != checksum:
            raise ValueError(f'Checksum mismatch: {name}')
    config = json.loads((directory/'config.json').read_text())
    rows = json.loads((directory/'results.json').read_text())['records']
    expected = 2*len(config['legendre_modes'])*len(config['schemes'])*2*config['repeats']
    if len(rows) != expected or manifest['completed_cases'] != expected:
        raise ValueError('Missing optimization cases')
    records = {(r['size'], r['scheme'], r['schedule'], r['repeat'], r['variant']): r for r in rows}
    if len(records) != expected:
        raise ValueError('Duplicate optimization case')
    pairs = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    with np.load(directory/'fields.npz') as fields:
        for row in rows:
            if row['variant'] != 'baseline':
                continue
            size, scheme, schedule, repeat = (row[key] for key in ('size', 'scheme', 'schedule', 'repeat'))
            other = records[size, scheme, schedule, repeat, 'optimized']
            first, second = row['field_key'], other['field_key']
            phi, pressure_basis, weight, _, _ = _quadrature(size, config['error_quadrature'],
                                                          config['lx'], config['ly'])
            velocity = [fields[key+'__velocity'] for key in (first, second)]
            difference = np.array([_physical(c, phi, size) for c in (velocity[1]-velocity[0]).reshape(2, -1)])
            reference = np.array([_physical(c, phi, size) for c in velocity[0].reshape(2, -1)])
            velocity_difference = _norm(difference, weight)/max(_norm(reference, weight), 1e-12)
            pressure = [fields[key+'__pressure'] for key in (first, second)]
            pressure_difference = 0.
            pressure_absolute_difference = pressure_scaled_error = 0.
            for p, q in zip(*pressure):
                value = _norm(_physical(q-p, pressure_basis, size), weight)
                scale = _norm(_physical(p, pressure_basis, size), weight)
                pressure_difference = max(pressure_difference, value/max(scale, 1e-12))
                pressure_absolute_difference = max(pressure_absolute_difference, value)
                tolerance = config.get('pressure_equivalence_absolute_tolerance', 0.) \
                    + config['equivalence_relative_gate']*scale
                pressure_scaled_error = max(pressure_scaled_error, value/max(tolerance, np.finfo(float).tiny))
            root_counts_equal = np.array_equal(fields[first+'__root_count'], fields[second+'__root_count'])
            roots_equal = np.allclose(fields[first+'__roots'], fields[second+'__roots'],
                                     rtol=1e-10, atol=1e-12, equal_nan=True)
            stage_r_difference = float(np.max(np.abs(fields[first+'__stage_r']-fields[second+'__stage_r'])))
            diagnostic_scale = np.maximum(1., np.abs(fields[first+'__diagnostics']))
            diagnostic_difference = float(np.max(np.abs(fields[first+'__diagnostics']-
                                                         fields[second+'__diagnostics'])/diagnostic_scale))
            times_equal = np.array_equal(fields[first+'__times'], fields[second+'__times'])
            passed = (max(velocity_difference, stage_r_difference, diagnostic_difference)
                      <= config['equivalence_relative_gate'] and pressure_scaled_error <= 1.
                      and root_counts_equal and roots_equal and times_equal)
            pair = {'size': size, 'scheme': scheme, 'schedule': schedule, 'repeat': repeat,
                    'velocity_relative_difference': velocity_difference,
                    'pressure_relative_difference': pressure_difference,
                    'pressure_absolute_difference': pressure_absolute_difference,
                    'pressure_equivalence_scaled_error': pressure_scaled_error,
                    'stage_r_absolute_difference': stage_r_difference,
                    'diagnostics_scaled_difference': diagnostic_difference,
                    'roots_equal': bool(roots_equal and root_counts_equal), 'times_equal': times_equal,
                    'equivalence_passed': bool(passed)}
            for record, coefficients in zip((row, other), velocity):
                error = _error(config, size, coefficients, record['final_time'], config['error_quadrature'])
                refined = _error(config, size, coefficients, record['final_time'], config['error_quadrature_refined'])
                sensitivity = abs(error-refined)/max(refined, np.finfo(float).tiny)
                pair[record['variant']+'_relative_L2'] = refined
                pair[record['variant']+'_quadrature_passed'] = bool(sensitivity <= config['error_quadrature_relative_gate'])
            pairs.append(pair)
            groups[size, scheme, schedule].append({'baseline': row, 'optimized': other, 'pair': pair})
    summaries = []
    for (size, scheme, schedule), members in sorted(groups.items()):
        summary = {'size': size, 'scheme': scheme, 'schedule': schedule, 'repeats': len(members),
                   'equivalence_passed': all(m['pair']['equivalence_passed'] for m in members),
                   'relative_L2': members[0]['pair']['optimized_relative_L2']}
        for variant in ('baseline', 'optimized'):
            for key in ('setup_cpu_seconds', 'integration_cpu_seconds', 'total_cpu_seconds', 'total_wall_seconds',
                        'assembly_and_factorization_wall_seconds', 'nonlinear_cpu_seconds',
                        'diagnostics_cpu_seconds', 'cached_stokes_cpu_seconds'):
                summary[variant+'_'+key] = median(m[variant][key] for m in members)
            summary[variant+'_factorizations'] = members[0][variant]['factorizations']
        summary['total_cpu_speedup'] = summary['baseline_total_cpu_seconds']/summary['optimized_total_cpu_seconds']
        summaries.append(summary)
    analysis = {'input_run': str(directory), 'input_manifest_sha256': digest(directory/'manifest.json'),
                'all_equivalence_passed': all(p['equivalence_passed'] for p in pairs),
                'all_error_quadrature_passed': all(p[v+'_quadrature_passed'] for p in pairs for v in ('baseline', 'optimized')),
                'summaries': summaries, 'pairs': pairs,
                'memory': [json.loads(path.read_text()) for path in sorted(directory.glob('memory_*.json'))]}
    report = root/'reports/legendre_optimization'/manifest['id']
    report.mkdir(parents=True, exist_ok=True)
    write_json(report/'analysis.json', analysis)
    text = ['# Legendre 优化对照', '',
            f"优化后端：{config.get('optimized_backend', 'direct')}。", '',
            '总CPU为初始化加完整推进；三次交错单线程重复取中位数。物理误差由保存系数独立重构。', '',
            '|模态|格式|步长|原总CPU/ms|优化总CPU/ms|加速|速度相对L2误差|',
            '|---:|---|---|---:|---:|---:|---:|']
    for row in summaries:
        text.append(f"|{row['size']}|{row['scheme']}|{row['schedule']}|"
                    f"{1000*row['baseline_total_cpu_seconds']:.3f}|{1000*row['optimized_total_cpu_seconds']:.3f}|"
                    f"{row['total_cpu_speedup']:.2f}|{row['relative_L2']:.3e}|")
    text += ['', f"全部数值等价检查通过：{analysis['all_equivalence_passed']}。",
             f"全部误差求积检查通过：{analysis['all_error_quadrature_passed']}。", '',
             'profiles.json 是独立插桩积分的函数调用及累计/自身墙钟时间，不能相加当作主计时占比。',
             '内存样本来自独立进程的最大模态、SDIRK3-mr-ccSAV、给定变步长；RSS包含解释器及导入。',
             '结论限定于本机、光滑制造解和这个短时窗口；给定变步长不含控制器及拒步成本。']
    (report/'comparison.md').write_text('\n'.join(text)+'\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print('REPORT='+str(analyze(args.run.resolve())))


if __name__ == '__main__':
    main()

"""Readonly comparison of explicitly saved cylinder runs; no flow model is constructed."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.workflow import PROJECT, digest, write_json
from experiments.cylinder_ns.run import load_cylinder
from experiments.cylinder_ns.representation import validate
from experiments.cylinder_ns.plots import _vorticity
from experiments.cylinder_ns.benchmark import cycle_statistics, reference_comparison


def read_case(path: Path) -> dict[str, Any]:
    """Own actual saved mesh/fields and preserve physical versus weak-divergence diagnostics."""
    cfg, manifest = load_cylinder(path, require_complete=True)
    if not cfg['scheme'].endswith('_mrsav'):
        raise ValueError('Cylinder wake comparison requires SAV results')
    with h5py.File(path/'results.h5', 'r') as data:
        basis, _ = validate(data, cfg, manifest)
        fields = {key: data['snapshots/'+key][:] for key in ('u', 'v', 'actual_times', 'requested_times')}
        history = {key: data['observations/'+key][:] for key in ('time', 'drag', 'lift', 'pressure_difference')}
        history['r'] = data['diagnostics/r'][:]
        maximum_r = float(np.max(np.abs(history['r'])))
        weak = float(np.max(data['diagnostics/continuity_residual'][:]))
        physical = float(np.max(data['diagnostics/divergence_inf'][:]))
        stage = float(np.max(data['stages/residual'][:], initial=0.))
    if maximum_r > .05 or weak > 1e-8 or stage > 1e-8:
        raise ValueError('SAV drift or algebraic residual gate failed; preserve raw run and inspect it')
    cycles = cycle_statistics(history['time'], history['drag'], history['lift'], history['pressure_difference'],
                              mean_velocity=2*cfg['inlet_max']/3, diameter=2*cfg['radius'])
    comparison = ({'applicable': True, **reference_comparison(cycles, cfg, complete=True)}
                  if cfg['reynolds'] == 100. else
                  {'applicable': False, 'passed': False, 'reason': 'The DFG2D-2 Re100 reference does not qualify a higher-Re wake'})
    return {'config': cfg, 'manifest': manifest, 'basis': basis, 'fields': fields, 'history': history,
            'cycles': cycles, 'reference': comparison,
            'metrics': {'max_abs_r': maximum_r, 'max_weak_continuity': weak,
                        'max_quadrature_physical_divergence': physical, 'max_stage_residual': stage}}


def analyze(directory: Path, *, root: Path = PROJECT, dpi: int = 100) -> Path:
    """Render both actual-time sequences with one fixed dimensionless-vorticity scale."""
    directory = directory.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    controls = json.loads((directory/'config.json').read_text())
    if manifest['status'] != 'complete' or digest(directory/'config.json') != manifest['config_sha256']:
        raise ValueError('Explicit complete cylinder comparison required')
    records = []
    for case in manifest['cases']:
        result = json.loads((Path(case['directory'])/'result.json').read_text())
        record = read_case(Path(result['run']))
        if record['manifest']['source']['code_sha256'] != manifest['preparation_source']['code_sha256']:
            raise ValueError('Numerical workers do not match the frozen comparison source')
        if record['manifest']['results_sha256'] != result['results_sha256']:
            raise ValueError('Cylinder worker input checksum mismatch')
        records.append(record)
    if len(records) != 2:
        raise ValueError('Exactly two cylinder runs are required')
    a, b = records
    for key in ('scheme', 'gamma', 'inlet_max', 'lx', 'ly', 'resolution', 'circle_segments', 'controller', 'scalar_atol',
                'boundary', 'physical_pressure_semantics'):
        if a['config'][key] != b['config'][key]: raise ValueError('Comparison controls differ: '+key)
    if not np.array_equal(a['basis'].mesh.p, b['basis'].mesh.p) or not np.array_equal(a['basis'].mesh.t, b['basis'].mesh.t):
        raise ValueError('Comparison must use identical saved meshes')
    for record in records:
        fields = record['fields']
        if len(fields['actual_times']) < 3 or np.any(np.diff(fields['actual_times']) <= 0):
            raise ValueError('Need distinct actual saved times')
        if not np.array_equal(fields['requested_times'], a['fields']['requested_times']):
            raise ValueError('Requested comparison snapshots differ')
    report = root/'reports'/'cylinder_wake_compare'/directory.name
    report.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {'status': 'rendering', 'input': str(directory),
        'source_manifest_sha256': digest(directory/'manifest.json'), 'render_source_sha256': digest(Path(__file__)),
        'cases': [{'run': str(Path(c['directory'])/'result.json'), 'config': r['config'],
            'metrics': r['metrics'], 'cycles': r['cycles'], 'reference': r['reference'],
            'actual_times': r['fields']['actual_times'].tolist()} for c, r in zip(manifest['cases'], records)],
        'time_semantics': 'paired requested nodes; each row clock shows its own actual accepted-node time; no temporal interpolation',
        'field_semantics': 'exact affine-P2 cell-average vorticity times D/Umean',
        'limits': 'Two-dimensional confined wakes. Higher Re does not imply chaos or three-dimensional turbulence. '
                  'Exploratory mesh; benchmark qualification and short checks are reported separately.'}
    if (directory/'checks.json').exists():
        if digest(directory/'checks.json') != manifest.get('checks_sha256'):
            raise ValueError('Sensitivity report checksum mismatch')
        checks = json.loads((directory/'checks.json').read_text())
        if digest(directory/'checks.npz') != checks['samples_sha256']:
            raise ValueError('Sensitivity samples checksum mismatch')
        with np.load(directory/'checks.npz', allow_pickle=False) as samples:
            for index, row in enumerate(checks['rows']):
                if row['results_sha256'] != records[index]['manifest']['results_sha256']:
                    raise ValueError('Sensitivity inputs differ from the saved comparison')
                w, base = samples[f'case_{index}_weights'], samples[f'case_{index}_base']
                for key, target in (('tight', 'time_relative_L2'), ('fine', 'space_relative_L2')):
                    other = samples[f'case_{index}_'+key]
                    value = float(np.sqrt(np.sum((base-other)**2*w)/np.sum(other**2*w)))
                    if not np.isclose(value, row[target], rtol=1e-12, atol=1e-14):
                        raise ValueError('Stored sensitivity norm mismatch')
        summary['short_sensitivity'] = checks
    write_json(report/'analysis.json', summary)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None or shutil.which('xelatex') is None or shutil.which('pdffonts') is None:
        raise RuntimeError('Movie export requires ffmpeg, XeLaTeX and pdffonts')
    import matplotlib as mpl
    import matplotlib.tri as mtri
    from matplotlib import font_manager
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon

    tex: Any = {'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 12,
        'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
        'pgf.preamble': r'\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{amsmath}'}
    try:
        with mpl.rc_context(tex):
            figure = Figure(figsize=(13.6, 10.), dpi=dpi, facecolor='#f8f9fb')
            canvas = FigureCanvasAgg(figure)
            gs = figure.add_gridspec(3, 2, left=.065, right=.91, bottom=.10, top=.80,
                                    height_ratios=[1.65, 1.65, 1.], hspace=.30, wspace=.3)
            flow_axes = [figure.add_subplot(gs[i, :]) for i in range(2)]
            lift_ax, scalar_ax = figure.add_subplot(gs[2, 0]), figure.add_subplot(gs[2, 1])
            images = []
            diameter = 2*a['config']['radius']
            mean = 2*a['config']['inlet_max']/3
            mesh = a['basis'].mesh
            center = np.array([a['config']['center_x'], a['config']['center_y']])
            coordinates = (mesh.p-center[:, None])/diameter
            triangles = mtri.Triangulation(coordinates[0], coordinates[1], mesh.t.T)
            assert mesh.boundaries is not None
            body = np.unique(mesh.facets[:, mesh.boundaries['cylinder']])
            angle = np.arctan2(coordinates[1, body], coordinates[0, body])
            polygon = coordinates[:, body[np.argsort(angle)]].T
            colours = ('#187d9c', '#c45438')
            for index, (axis, record) in enumerate(zip(flow_axes, records)):
                image = axis.tripcolor(triangles, facecolors=np.zeros(mesh.t.shape[1]), cmap='RdBu_r',
                                      vmin=-controls['vorticity_limit'], vmax=controls['vorticity_limit'])
                images.append(image)
                axis.add_patch(Polygon(polygon, closed=True, facecolor='#f8f9fb', edgecolor='#172536', lw=1.2))
                label = ('periodic shedding' if record['cycles'].get('developed') else 'evolving wake') if index == 0 else '2D unsteady wake'
                axis.set(xlim=(-2., 20.), ylim=(-2., 2.1), xlabel=r'$(x-x_c)/D$', ylabel=r'$(y-y_c)/D$')
                axis.set_aspect('equal', adjustable='box')
                axis.set_title(rf'$Re={record["config"]["reynolds"]:g}$: '+label, loc='left', pad=9)
            top, bottom = (axis.get_position() for axis in flow_axes)
            for position in (top, bottom):
                if not np.isclose(position.width*13.6/22., position.height*10./4.1):
                    raise RuntimeError('Cylinder display must preserve equal physical axis scales')
            summary['flow_axes_use_equal_scale'] = True
            cax = figure.add_axes((.93, bottom.y0, .013, top.y1-bottom.y0))
            figure.colorbar(images[0], cax=cax, label=r'$\omega D/U_{mean}$', extend='both')
            selected = [np.searchsorted(r['history']['time'], controls['display_start']) for r in records]
            for index, record in enumerate(records):
                lift_ax.plot([], [], color=colours[index], label=rf'$Re={record["config"]["reynolds"]:g}$')
                scalar_ax.plot([], [], color=colours[index], label=rf'$Re={record["config"]["reynolds"]:g}$')
            lift_limit = max(float(np.max(np.abs(r['history']['lift'][s:]))) for r, s in zip(records, selected))
            scalar_limit = max(1e-9, max(float(np.max(np.abs(r['history']['r'][s:]))) for r, s in zip(records, selected)))
            lift_ax.set(xlim=(controls['display_start'], controls['T']), ylim=(-1.15*lift_limit, 1.15*lift_limit),
                        xlabel='$t$', ylabel='$C_L$', title='Physical lift history')
            scalar_ax.set(xlim=(controls['display_start'], controls['T']), ylim=(-1.15*scalar_limit, 1.15*scalar_limit),
                          xlabel='$t$', ylabel='$r$', title='SAV scalar history')
            for axis in (lift_ax, scalar_ax): axis.grid(alpha=.2); axis.legend(loc='upper right', fontsize=9)
            figure.text(.065, .945, 'Cylinder wakes: a two-dimensional SAV comparison', fontsize=24, weight='bold', color='#172536')
            figure.text(.065, .895, r'$Re=U_{mean}D/\nu$  $\cdot$  '+a['config']['scheme'].replace('_', '-')
                        +rf'  $\cdot$  $\gamma={a["config"]["gamma"]:g}$', fontsize=13, color='#526275')
            outlet_label = ('backflow-damped outlet' if a['config'].get('outlet_treatment') == 'backflow_damped'
                            else 'natural outlet')
            figure.text(.065, .85, 'Same saved mesh, cylinder and parabolic inlet; no-slip walls; '+outlet_label+'.', fontsize=11, color='#526275')
            figure.text(.065, .035, 'Exploratory mesh; short sensitivity checks only. Higher-Re panel: 2D unsteady wake, not 3D turbulence.', fontsize=10, color='#526275')
            for image in images: image.set_visible(False)
            figure.savefig(report/'layout.pdf', backend='pgf')
            figure.savefig(report/'layout.png', backend='pgf', dpi=dpi)
            probe = subprocess.run(['pdffonts', str(report/'layout.pdf')], check=True, capture_output=True, text=True)
            (report/'fonts.txt').write_text(probe.stdout)
            if 'TimesNewRoman' not in probe.stdout.replace('-', '').replace(' ', ''): raise RuntimeError('Font verification failed')
            background = Image.open(report/'layout.png').convert('RGBA')
            figure.patch.set_alpha(0.)
            for axis in figure.axes:
                axis.patch.set_alpha(0.); axis.xaxis.set_visible(False); axis.yaxis.set_visible(False)
                for loc in ('left', 'center', 'right'): axis.set_title('', loc=loc)
                for spine in axis.spines.values(): spine.set_visible(False)
                for artist in (*axis.lines, *axis.collections, *axis.texts): artist.set_visible(False)
                legend = axis.get_legend()
                if legend is not None: legend.set_visible(False)
            for text in figure.texts: text.set_visible(False)
            for image in images: image.set_visible(True)
            lift_lines = [lift_ax.plot([], [], color=colour, lw=1.5)[0] for colour in colours]
            scalar_lines = [scalar_ax.plot([], [], color=colour, lw=1.5)[0] for colour in colours]
            clock_font = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),
                                                                 fallback_to_default=False), int(dpi*.20))
            width, height = background.size
            command = [ffmpeg, '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                '-framerate', str(controls['fps']), '-i', '-', '-an', '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(report/'cylinder_wakes.mp4')]
            with (report/'ffmpeg.log').open('w') as log:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
                assert process.stdin is not None
                count = len(a['fields']['actual_times'])
                try:
                    for frame_index in range(count):
                        actual_times = []
                        for index, record in enumerate(records):
                            fields, history = record['fields'], record['history']
                            time = float(fields['actual_times'][frame_index]); actual_times.append(time)
                            images[index].set_array(_vorticity(record['basis'], fields['u'][frame_index], fields['v'][frame_index])*diameter/mean)
                            end = np.searchsorted(history['time'], time, side='right')
                            start = selected[index]
                            lift_lines[index].set_data(history['time'][start:end], history['lift'][start:end])
                            scalar_lines[index].set_data(history['time'][start:end], history['r'][start:end])
                        canvas.draw()
                        frame = Image.alpha_composite(background, Image.fromarray(np.asarray(canvas.buffer_rgba()))).convert('RGB')
                        draw = ImageDraw.Draw(frame)
                        for index, time in enumerate(actual_times):
                            draw.text((int(.9*width), int((1-flow_axes[index].get_position().y1-.032)*height)),
                                      f't = {time:.4f}', font=clock_font, fill='#172536', anchor='ra')
                        process.stdin.write(frame.tobytes())
                        if frame_index in {0, count//2, count-1}: frame.save(report/f'frame_{frame_index:04d}.png')
                finally:
                    process.stdin.close(); code = process.wait()
                if code: raise RuntimeError('ffmpeg failed; see ffmpeg.log')
            for name, scale, fps in (('cylinder_wakes.gif', 960, 15), ('preview.gif', 720, 10)):
                filters = f'[0:v]fps={fps},scale={scale}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a'
                subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', str(report/'cylinder_wakes.mp4'),
                                '-filter_complex', filters, '-loop', '0', str(report/name)], check=True)
            summary.update(status='complete', frames=count, fps=controls['fps'], duration_seconds=count/controls['fps'],
                font_verified=True, outputs={name: digest(report/name) for name in
                    ('cylinder_wakes.mp4', 'cylinder_wakes.gif', 'preview.gif', 'layout.pdf', 'layout.png')})
    except Exception as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(report/'analysis.json', summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args()
    print(analyze(args.run, root=args.root))


if __name__ == '__main__': main()

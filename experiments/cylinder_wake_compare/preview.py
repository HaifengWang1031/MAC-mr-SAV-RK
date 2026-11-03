"""Readonly single-run preview, explicitly separate from the controlled two-run comparison."""
import argparse
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.records import PROJECT, digest, write_json
from experiments.cylinder_ns.plots import _vorticity
from .analyze import read_case
from .smooth import SmoothVorticityGrid


def preview(directory: Path, *, root: Path = PROJECT, start: float = 10., fps: int = 15,
            dpi: int = 100, smooth: bool = False, smoothing_sigma: float = .008) -> Path:
    """Render actual saved fields and force history without constructing a flow model."""
    directory = directory.resolve()
    before = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    record = read_case(directory)
    cfg, fields, history, basis = (record[key] for key in ('config', 'fields', 'history', 'basis'))
    times, indices = np.unique(fields['actual_times'], return_index=True)
    # Select requested output nodes; their nearest accepted state may be just
    # before the requested window. Keep its true clock instead of dropping it.
    indices = indices[fields['requested_times'][indices] >= start]
    times = fields['actual_times'][indices]
    if len(indices) < 3 or fps < 1:
        raise ValueError('Need at least three saved preview frames and positive fps')
    family = 'cylinder_wake_smooth' if smooth else 'cylinder_wake_preview'
    report = root/'reports'/family/directory.name
    if report.resolve().is_relative_to(directory):
        raise ValueError('Preview reports must be separate from raw runs')
    report.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {'status': 'rendering', 'run': str(directory), 'input_fingerprints': before,
        'producer': record['manifest']['source'], 'config': cfg, 'metrics': record['metrics'],
        'cycles': record['cycles'], 'reference': record['reference'], 'actual_times': times.tolist(),
        'source_sha256': digest(Path(__file__)), 'field': 'affine P2 cell-average vorticity times D/Umean',
        'limits': 'Single saved exploratory run; this preview does not establish a completed two-Re comparison. '
                  'No temporal interpolation; two-dimensional wake only.'}
    write_json(report/'analysis.json', summary)
    if any(shutil.which(name) is None for name in ('ffmpeg', 'xelatex', 'pdffonts')):
        raise RuntimeError('Preview requires ffmpeg, XeLaTeX and pdffonts')
    import matplotlib as mpl
    import matplotlib.tri as mtri
    from matplotlib import font_manager
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon
    settings: Any = {'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 12,
        'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
        'pgf.preamble': r'\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{amsmath}'}
    try:
        with mpl.rc_context(settings):
            figure = Figure(figsize=(12.8, 6.), dpi=dpi, facecolor='#f8f9fb')
            canvas = FigureCanvasAgg(figure)
            axis, lift = figure.add_axes((.065, .42, .83, .38)), figure.add_axes((.065, .13, .83, .20))
            diameter, mean = 2*cfg['radius'], 2*cfg['inlet_max']/3
            xy = (basis.mesh.p-np.array([cfg['center_x'], cfg['center_y']])[:, None])/diameter
            display = None
            image: Any
            if smooth:
                display = SmoothVorticityGrid(basis, bounds=(0., cfg['lx'], 0., cfg['ly']), sigma=smoothing_sigma)
                summary['visualization'] = {**display.metadata(),
                    'source_sha256': digest(Path(__file__).with_name('smooth.py'))}
                summary['field'] = 'spatially smoothed recovered vorticity times D/Umean; display only'
                extent = (-cfg['center_x']/diameter, (cfg['lx']-cfg['center_x'])/diameter,
                          -cfg['center_y']/diameter, (cfg['ly']-cfg['center_y'])/diameter)
                image = axis.imshow(np.ma.array(np.zeros(display.mask.shape), mask=display.mask),
                    origin='lower', extent=extent, interpolation='bilinear', cmap='RdBu_r', vmin=-5., vmax=5.)
            else:
                triangle = mtri.Triangulation(xy[0], xy[1], basis.mesh.t.T)
                image = axis.tripcolor(triangle, facecolors=np.zeros(basis.mesh.t.shape[1]),
                                       cmap='RdBu_r', vmin=-5., vmax=5.)
            assert basis.mesh.boundaries is not None
            body = np.unique(basis.mesh.facets[:, basis.mesh.boundaries['cylinder']])
            body = body[np.argsort(np.arctan2(xy[1, body], xy[0, body]))]
            axis.add_patch(Polygon(xy[:, body].T, facecolor='#f8f9fb', edgecolor='#172536', lw=1.2))
            axis.set(xlim=(-2., 20.), ylim=(-2., 2.1), xlabel=r'$(x-x_c)/D$', ylabel=r'$(y-y_c)/D$')
            axis.set_aspect('equal', adjustable='box')
            label = ('2D unsteady wake' if cfg['reynolds'] > 100. else
                     'periodic vortex shedding' if record['cycles'].get('developed') else 'evolving wake')
            axis.set_title(rf'$Re={cfg["reynolds"]:g}$: '+label, loc='left', pad=9)
            pos = axis.get_position()
            if not np.isclose(pos.width*12.8/22., pos.height*6./4.1):
                raise RuntimeError('Cylinder display aspect ratio changed')
            cax = figure.add_axes((.92, pos.y0, .015, pos.height))
            color_label = r'$\widetilde{\omega}D/U_{mean}$' if smooth else r'$\omega D/U_{mean}$'
            figure.colorbar(image, cax=cax, label=color_label, extend='both')
            selected = history['time'] >= times[0]
            limit = max(.01, float(np.max(np.abs(history['lift'][selected]))))*1.15
            lift.set(xlim=(times[0], times[-1]), ylim=(-limit, limit), xlabel='$t$', ylabel='$C_L$')
            if not smooth:
                lift.grid(alpha=.2)
            heading = 'Cylinder wake: smooth visualization' if smooth else 'Cylinder wake: saved SAV result'
            figure.text(.065, .94, heading, fontsize=23, weight='bold', color='#172536')
            mantissa, exponent = f'{record["metrics"]["max_abs_r"]:.3e}'.split('e')
            scalar_label = rf'{mantissa}\times10^{{{int(exponent)}}}'
            figure.text(.065, .875, cfg['scheme'].replace('_', '-')+rf'  $\cdot$  $\gamma={cfg["gamma"]:g}$'
                        +rf'  $\cdot$  $\max|r|={scalar_label}$', color='#526275')
            outlet = ('backflow-damped outlet' if cfg.get('outlet_treatment') == 'backflow_damped'
                      else 'natural outlet')
            figure.text(.065, .835, 'No-slip walls and cylinder; '+outlet+'.', fontsize=11, color='#526275')
            qualification = ('passed' if record['reference']['passed'] else 'NOT PASSED') if cfg['reynolds'] == 100. else 'not applicable'
            footer = ('Spatially smoothed for display only; spatial accuracy remains unverified.' if smooth else
                      'Exploratory saved mesh; Re100 reference gate: '+qualification+
                      '. Single-run preview; no paired-Re result shown.')
            figure.text(.065, .035, footer, fontsize=10, color='#526275')
            image.set_visible(False)
            figure.savefig(report/'layout.pdf', backend='pgf')
            figure.savefig(report/'layout.png', backend='pgf', dpi=dpi)
            fonts = subprocess.check_output(['pdffonts', str(report/'layout.pdf')], text=True)
            (report/'fonts.txt').write_text(fonts)
            if 'TimesNewRoman' not in fonts.replace('-', '').replace(' ', ''):
                raise RuntimeError('Font verification failed')
            background = Image.open(report/'layout.png').convert('RGBA')
            figure.patch.set_alpha(0.)
            for ax in figure.axes:
                ax.patch.set_alpha(0.); ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
                for loc in ('left', 'center', 'right'): ax.set_title('', loc=loc)
                for spine in ax.spines.values(): spine.set_visible(False)
                for artist in (*ax.lines, *ax.collections, *ax.texts): artist.set_visible(False)
            for text in figure.texts: text.set_visible(False)
            image.set_visible(True)
            line, = lift.plot([], [], color='#187d9c', lw=1.5)
            font = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),
                                                           fallback_to_default=False), int(dpi*.20))
            width, height = background.size
            with (report/'ffmpeg.log').open('w') as log:
                command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                    '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                    '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(report/'wake.mp4')]
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
                assert process.stdin is not None
                try:
                    for k, (index, time) in enumerate(zip(indices, times)):
                        if display is None:
                            image.set_array(_vorticity(basis, fields['u'][index], fields['v'][index])*diameter/mean)
                        else:
                            image.set_data(display.frame(fields['u'][index], fields['v'][index])*diameter/mean)
                        mask = (history['time'] >= times[0]) & (history['time'] <= time)
                        line.set_data(history['time'][mask], history['lift'][mask])
                        canvas.draw()
                        frame = Image.alpha_composite(background, Image.fromarray(np.asarray(canvas.buffer_rgba()))).convert('RGB')
                        ImageDraw.Draw(frame).text((int(.895*width), int((1-pos.y1-.04)*height)),
                            f't = {time:.5f}', font=font, fill='#172536', anchor='ra')
                        process.stdin.write(frame.tobytes())
                        if k in {0, len(indices)//2, len(indices)-1}: frame.save(report/f'frame_{k:04d}.png')
                finally:
                    process.stdin.close(); code = process.wait()
                if code: raise RuntimeError('ffmpeg preview failed; inspect ffmpeg.log')
            for name, scale in (('wake.gif', 960), ('preview.gif', 720)):
                filters = f'[0:v]scale={scale}:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse'
                subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(report/'wake.mp4'),
                                '-filter_complex', filters, '-loop', '0', str(report/name)], check=True)
            if before != {p.name: digest(p) for p in directory.iterdir() if p.is_file()}:
                raise RuntimeError('Readonly preview changed raw input')
            summary.update(status='complete', frames=len(indices), fps=fps, font_verified=True,
                           flow_axes_use_equal_scale=True, outputs={name: digest(report/name) for name in
                           ('wake.mp4', 'wake.gif', 'preview.gif', 'layout.pdf', 'layout.png')})
    except Exception as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(report/'analysis.json', summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--start', type=float, default=10.)
    parser.add_argument('--smooth', action='store_true', help='continuous, spatially smoothed visualization in a separate report')
    parser.add_argument('--smoothing-sigma', type=float, default=.008, help='display Gaussian width in physical length units')
    args = parser.parse_args()
    print(preview(args.run, start=args.start, smooth=args.smooth, smoothing_sigma=args.smoothing_sigma))


if __name__ == '__main__': main()

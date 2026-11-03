"""Stream saved wedge-cavity frames to video with one static, verified layout."""
from pathlib import Path
from time import perf_counter
from typing import Any
import json
import subprocess

import matplotlib as mpl
import matplotlib.tri as mtri
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

from experiments.records import digest
from .fields import FieldSampler, streamline_levels


def render(directory: Path, *, output: Path, fps: int = 24, dpi: int = 165,
           step: int = 1, hold: int = 12, sampler: FieldSampler | None = None,
           progress_every: int = 50, keep_frames: bool = False,
           detail: str = 'vorticity') -> dict[str, Any]:
    """Reuse artists and encode directly; no flow integration or PNG round trips."""
    from .report import open_run, _style_axes

    if min(fps, dpi, step) < 1 or hold < 0:
        raise ValueError('Positive fps, dpi, step and nonnegative hold required')
    if detail not in ('vorticity', 'lower_cell'):
        raise ValueError('detail must be vorticity or lower_cell')
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.is_relative_to(directory):
        raise ValueError('Animation output must be separate from raw runs')
    output.parent.mkdir(parents=True, exist_ok=True)
    before = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    cfg, manifest, space, model, data = open_run(directory)
    sampler = sampler or FieldSampler(model)
    picked = list(range(0, len(data.snapshot_times), step))
    if picked[-1] != len(data.snapshot_times)-1:
        picked.append(len(data.snapshot_times)-1)
    final = sampler.sample(data.state(picked[-1]))
    speed_limit = float(np.max(np.hypot(data.u, data.v)))/cfg['lid_speed']
    omega_scale = space.height/cfg['lid_speed']
    omega_limit = max(1e-12, float(np.percentile(np.abs(final.vorticity*omega_scale), 88.)))
    psi_levels = streamline_levels(final.streamfunction)
    primary_y = float(final.y[np.argmin(final.streamfunction)])
    lower = final.y < min(primary_y-.15*cfg['lid_length'], -.2*space.height)
    candidates = np.flatnonzero(lower)
    lower_index = int(candidates[np.argmax(final.streamfunction[lower])]) if len(candidates) else 0
    lower_peak = max(0., float(final.streamfunction[lower_index])) if len(candidates) else 0.
    center_y = float(final.y[lower_index]) if lower_peak > 1e-12 else -.6*space.height
    half_width = max(.04*cfg['lid_length'], space.half_width*(1.+center_y/space.height))
    radius_y = max(.08*cfg['lid_length'], 1.8*half_width)
    lower_bounds = (-1.25*half_width, 1.25*half_width,
                    max(-space.height, center_y-radius_y), min(0., center_y+radius_y))
    psi_scale = 1./(cfg['lid_speed']*cfg['lid_length'])
    lower_limit = max(1e-12, lower_peak*psi_scale)
    summary: dict[str, Any] = {
        'status': 'rendering', 'run_id': directory.name, 'run': str(directory),
        'input_fingerprints': before, 'producer': manifest['source'], 'config': cfg,
        'fps': fps, 'step': step, 'hold_frames': hold,
        'actual_times': data.snapshot_times[picked].tolist(), 'temporal_interpolation': False,
        'speed_limits': [0., speed_limit], 'vorticity_limits': [-omega_limit, omega_limit],
        'vorticity_units': 'omega H / U0', 'source_sha256': digest(Path(__file__)),
        'detail': detail, 'lower_region_bounds': list(lower_bounds),
        'lower_positive_streamfunction_candidate': lower_peak > 1e-12,
        'lower_streamfunction_color_limits': [-lower_limit, lower_limit],
        'limits': 'Startup visualization, not a steady-state, spatial-convergence or PDE-order qualification. '
                  'Streamfunction is a Poisson recovery; stage pressure is not plotted as physical pressure.',
    }
    record = output.with_suffix('.json')
    record.write_text(json.dumps(summary, indent=2)+'\n')
    settings: Any = {
        'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 12,
        'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
        'pgf.preamble': r'\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{amsmath}',
    }
    started = perf_counter()
    try:
        with mpl.rc_context(settings):
            figure = Figure(figsize=(11.4, 4.7), dpi=dpi, facecolor='#f8f9fb')
            canvas = FigureCanvasAgg(figure)
            axes = [figure.add_axes((left, .15, .36, .64)) for left in (.055, .555)]
            triangle = mtri.Triangulation(final.x, final.y, final.triangles)
            images = []
            for index, axis in enumerate(axes):
                if detail == 'lower_cell':
                    norm = (mpl.colors.LogNorm(vmin=min(1e-6, speed_limit/10), vmax=speed_limit)
                            if index == 0 else mpl.colors.Normalize(vmin=-lower_limit, vmax=lower_limit))
                else:
                    norm = mpl.colors.Normalize(vmin=0. if index == 0 else -omega_limit,
                                                vmax=speed_limit if index == 0 else omega_limit)
                image = axis.tripcolor(triangle, np.zeros_like(final.speed), shading='gouraud',
                    cmap='viridis' if index == 0 else 'RdBu_r', norm=norm, rasterized=True)
                images.append(image)
                _style_axes(axis, space)
                near_lid = detail == 'vorticity' and index == 1 and space.height > cfg['lid_length']
                if near_lid:
                    axis.set_ylim(-.8*cfg['lid_length'], .05*cfg['lid_length'])
                title = 'Speed and recovered streamlines' if index == 0 else (
                    'Near-lid vorticity and streamlines' if near_lid else 'Vorticity and streamlines')
                if detail == 'lower_cell':
                    title = 'Full cavity: log speed and streamlines' if index == 0 else 'Lower circulation region'
                    if index == 1:
                        axis.set(xlim=lower_bounds[:2], ylim=lower_bounds[2:])
                axis.set_title(title, color='#20364e', fontsize=12, pad=8)
                bounds = axis.get_position()
                color_axis = figure.add_axes((bounds.x1+.015, bounds.y0, .012, bounds.height))
                figure.colorbar(image, cax=color_axis, extend='max' if index == 0 else 'both',
                    label=r'$|\mathbf{u}|/U_0$' if index == 0 else (
                        r'$\psi/(U_0L)$' if detail == 'lower_cell' else r'$\omega H/U_0$'))
            figure.text(.055, .945, 'Lid-driven wedge cavity', fontsize=22, color='#18344e')
            figure.text(.055, .865,
                rf'$2\alpha={cfg["apex_angle"]:g}^\circ$, $Re={cfg["reynolds"]:g}$'
                + '  |  '+cfg['scheme'].replace('sdirk', 'SDIRK').replace('_mrsav', '-mr-ccSAV'),
                fontsize=13, color='#526275')
            figure.text(.055, .045, 'Moving lid to the right; stationary inclined walls. Saved fields, no time interpolation.',
                        fontsize=10, color='#526275')
            for image in images:
                image.set_visible(False)
            layout = output.with_name(output.stem+'_layout')
            figure.savefig(layout.with_suffix('.pdf'), backend='pgf')
            figure.savefig(layout.with_suffix('.png'), backend='pgf', dpi=dpi)
            fonts = subprocess.check_output(['pdffonts', str(layout.with_suffix('.pdf'))], text=True)
            output.with_name(output.stem+'_fonts.txt').write_text(fonts)
            if 'TimesNewRoman' not in fonts.replace('-', '').replace(' ', ''):
                raise RuntimeError('Times New Roman export verification failed')
            background = Image.open(layout.with_suffix('.png')).convert('RGBA')
            label_size = background.size
            if background.size != canvas.get_width_height():
                # PGF rounds raster dimensions up; Agg rounds down. Align only
                # the static label layer, leaving all sampled field data intact.
                background = background.resize(canvas.get_width_height(), Image.Resampling.LANCZOS)
            summary['static_label_raster_sizes'] = [list(label_size), list(background.size)]
            figure.patch.set_alpha(0.)
            for axis in figure.axes:
                axis.patch.set_alpha(0.)
                axis.xaxis.set_visible(False)
                axis.yaxis.set_visible(False)
                for location in ('left', 'center', 'right'):
                    axis.set_title('', loc=location)
                for spine in axis.spines.values():
                    spine.set_visible(False)
                for artist in (*axis.collections, *axis.texts):
                    artist.set_visible(False)
            for text in figure.texts:
                text.set_visible(False)
            for image in images:
                image.set_visible(True)
            font = ImageFont.truetype(font_manager.findfont(
                font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False), round(.16*dpi))
            width, height = background.size
            frame_dir = output.with_name(output.stem+'_frames')
            if keep_frames:
                frame_dir.mkdir(exist_ok=True)
            command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', '-', '-an',
                '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', str(output)]
            contours: list[Any] = []
            with output.with_name(output.stem+'_ffmpeg.log').open('w') as log:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
                assert process.stdin is not None
                try:
                    last_frame = b''
                    for position, index in enumerate(picked):
                        sample = sampler.sample(data.state(index))
                        images[0].set_array(sample.speed/cfg['lid_speed'])
                        images[1].set_array(sample.streamfunction*psi_scale if detail == 'lower_cell'
                                            else sample.vorticity*omega_scale)
                        for contour in contours:
                            contour.remove()
                        contours.clear()
                        active = psi_levels[(psi_levels > np.min(sample.streamfunction))
                                            & (psi_levels < np.max(sample.streamfunction))]
                        if len(active):
                            for axis, color in zip(axes, ('#f5f5f0', '#24364a')):
                                contours.append(axis.tricontour(triangle, sample.streamfunction,
                                    levels=active, colors=color, linewidths=.65, alpha=.65))
                        canvas.draw()
                        frame = Image.alpha_composite(background, Image.fromarray(np.asarray(canvas.buffer_rgba()))).convert('RGB')
                        ImageDraw.Draw(frame).text((round(.95*width), round(.135*height)),
                            f't = {data.snapshot_times[index]:.3f}', font=font, fill='#18344e', anchor='ra')
                        last_frame = frame.tobytes()
                        process.stdin.write(last_frame)
                        if keep_frames:
                            frame.save(frame_dir/f'frame_{position:05d}.png')
                        if position in {0, len(picked)//2, len(picked)-1}:
                            frame.save(output.with_name(output.stem+f'_frame_{position:04d}.png'))
                        if progress_every and (position+1) % progress_every == 0:
                            print(f'frame {position+1}/{len(picked)}: {perf_counter()-started:.1f}s', flush=True)
                    for _ in range(hold):
                        process.stdin.write(last_frame)
                finally:
                    process.stdin.close()
                    code = process.wait()
                if code:
                    raise RuntimeError('Animation encoding failed; inspect ffmpeg log')
            video_seconds = perf_counter()-started
            gif = output.with_suffix('.gif')
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(output), '-filter_complex',
                '[0:v]scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                '-loop', '0', str(gif)], check=True)
            if before != {p.name: digest(p) for p in directory.iterdir() if p.is_file()}:
                raise RuntimeError('Animation changed a raw input file')
            summary.update(status='complete', video=str(output), gif=str(gif), frames=len(picked),
                snapshots_used=len(picked), duration_seconds=(len(picked)+hold)/fps,
                t_range=[float(data.snapshot_times[picked[0]]), float(data.snapshot_times[picked[-1]])],
                rendered_seconds=video_seconds, total_export_seconds=perf_counter()-started,
                font_verified=True, inputs_unchanged=True,
                frames_dir=str(frame_dir) if keep_frames else None,
                near_lid_zoom=space.height > cfg['lid_length'])
    except Exception as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.write_text(json.dumps(summary, indent=2)+'\n')
    return summary

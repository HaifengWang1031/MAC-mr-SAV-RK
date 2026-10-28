"""Animate an explicit saved lid-driven cavity run without advancing the PDE.

Fixed scales show startup speed, streamfunction contours and interior vorticity.
Tracers use bilinear MAC-face interpolation and RK2 in interpolated saved time;
they are visual trajectories, not a computed concentration or dye equation.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
from uuid import uuid4

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.workflow import PROJECT, digest, load_record, write_json
from solver.mac.grid import Array, MACGrid


@dataclass
class MovieData:
    config: dict[str, Any]
    manifest: dict[str, Any]
    times: Array
    requested_times: Array
    u: Array
    v: Array
    speed: Array
    streamfunction: Array
    vorticity: Array
    max_divergence: float
    max_stage_residual: float


def read_movie(directory: Path) -> MovieData:
    """Read a complete checksum-verified cavity record at its actual snapshot times."""
    cfg, manifest = load_record(directory, require_complete=True)
    if cfg.get('experiment') != 'cavity':
        raise ValueError('A lid-driven cavity run is required')
    grid = MACGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    with h5py.File(directory/'results.h5', 'r') as saved:
        times = saved['snapshots/actual_times'][:]
        requested = saved['snapshots/requested_times'][:]
        if len(times) < 2 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
            raise ValueError('Animation needs distinct increasing snapshot times')
        if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-9, atol=1e-12):
            raise ValueError('Uniform actual snapshot spacing is required')
        if saved.attrs['status'] != 'complete':
            raise ValueError('Incomplete HDF5 record')
        u, v = saved['snapshots/u'][:], saved['snapshots/v'][:]
        residual = float(np.max(saved['stages/residual'][:], initial=0.))
        divergence = float(np.max(saved['diagnostics/divergence_inf'][:], initial=0.))
    if u.shape != (len(times), grid.ny, grid.nx+1) or v.shape != (len(times), grid.ny+1, grid.nx):
        raise ValueError('Invalid MAC snapshot shapes')
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError('Nonfinite saved velocity')
    for uu, vv in zip(u, v):
        grid.pack(uu, vv)
    actual_divergence = np.diff(u, axis=2)/grid.hx+np.diff(v, axis=1)/grid.hy
    if np.max(np.abs(actual_divergence)) > cfg['stokes_tolerance']*(1+max(np.abs(u).max(), np.abs(v).max())):
        raise ValueError('Saved snapshot full divergence check failed')
    uc, vc = .5*(u[:, :, :-1]+u[:, :, 1:]), .5*(v[:, :-1]+v[:, 1:])
    psi = np.concatenate([np.zeros((len(times), 1, grid.nx+1)), grid.hy*np.cumsum(u, axis=1)], axis=1)
    omega = np.diff(v[:, 1:-1], axis=2)/grid.hx-np.diff(u[:, :, 1:-1], axis=1)/grid.hy
    return MovieData(cfg, manifest, times, requested, u, v, np.hypot(uc, vc), psi, omega,
                     max(divergence, float(np.max(np.abs(actual_divergence)))), residual)


def sample_velocity(grid: MACGrid, u: Array, v: Array, points: Array, lid_speed: float) -> Array:
    """Bilinear MAC interpolation with stationary side/bottom and moving top wall."""
    top = np.full((1, grid.nx+1), lid_speed)
    top[0, [0, -1]] = 0.  # The sharp-lid corner value is not a velocity unknown.
    u_faces = np.vstack([np.zeros((1, grid.nx+1)), u, top])
    v_faces = np.column_stack([np.zeros(grid.ny+1), v, np.zeros(grid.ny+1)])
    x, y = np.clip(points[:, 0], 0., grid.lx), np.clip(points[:, 1], 0., grid.ly)
    result = np.empty_like(points, dtype=float)
    for component, faces, xs, ys in (
        (0, u_faces, np.arange(grid.nx+1)*grid.hx,
         np.r_[0., (np.arange(grid.ny)+.5)*grid.hy, grid.ly]),
        (1, v_faces, np.r_[0., (np.arange(grid.nx)+.5)*grid.hx, grid.lx],
         np.arange(grid.ny+1)*grid.hy),
    ):
        i = np.clip(np.searchsorted(xs, x, side='right')-1, 0, len(xs)-2)
        j = np.clip(np.searchsorted(ys, y, side='right')-1, 0, len(ys)-2)
        fx, fy = (x-xs[i])/(xs[i+1]-xs[i]), (y-ys[j])/(ys[j+1]-ys[j])
        bottom = (1-fx)*faces[j, i]+fx*faces[j, i+1]
        top_value = (1-fx)*faces[j+1, i]+fx*faces[j+1, i+1]
        result[:, component] = (1-fy)*bottom+fy*top_value
    return result


def tracer_paths(data: MovieData, *, count: int = 500, seed: int = 20261010) -> tuple[Array, int]:
    """Midpoint trajectories with <=0.005 time substeps and a reproducible seed."""
    if count < 1:
        raise ValueError('Tracer count must be positive')
    cfg = data.config
    grid = MACGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    rng = np.random.default_rng(seed)
    positions = rng.uniform(.025, .975, (count, 2))*np.array([grid.lx, grid.ly])
    paths = np.empty((len(data.times), count, 2))
    paths[0] = positions
    clipped = 0
    for index, interval in enumerate(np.diff(data.times)):
        count_steps = max(1, int(np.ceil(interval/.005)))
        step = interval/count_steps
        for j in range(count_steps):
            fraction = j/count_steps
            u = (1-fraction)*data.u[index]+fraction*data.u[index+1]
            v = (1-fraction)*data.v[index]+fraction*data.v[index+1]
            midpoint = positions+.5*step*sample_velocity(grid, u, v, positions, cfg['lid_speed'])
            fraction = (j+.5)/count_steps
            u = (1-fraction)*data.u[index]+fraction*data.u[index+1]
            v = (1-fraction)*data.v[index]+fraction*data.v[index+1]
            positions += step*sample_velocity(grid, u, v, midpoint, cfg['lid_speed'])
            outside = np.any((positions < 0.) | (positions > [grid.lx, grid.ly]), axis=1)
            clipped += int(np.count_nonzero(outside))
            positions = np.clip(positions, 0., [grid.lx, grid.ly])
        paths[index+1] = positions
    return paths, clipped


def render(directory: Path, *, root: Path = PROJECT, fps: int = 20,
           particles: int = 500, seed: int = 20261010, dpi: int = 100) -> Path:
    """Export MP4/GIF and preview frames using saved data and verified Times New Roman."""
    if type(fps) is not int or fps < 1 or type(dpi) is not int or dpi < 60:
        raise ValueError('Require positive fps and dpi >= 60')
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None or shutil.which('xelatex') is None or shutil.which('pdffonts') is None:
        raise RuntimeError('Export requires ffmpeg, XeLaTeX and pdffonts')
    import matplotlib as mpl
    from matplotlib import font_manager
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    directory = directory.resolve()
    data = read_movie(directory)
    cfg = data.config
    grid = MACGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    report = root/'reports'/'cavity_animation'/(directory.name+'-'+
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:6])
    report.mkdir(parents=True)
    summary: dict[str, Any] = {'status': 'rendering', 'run': str(directory), 'run_id': directory.name,
        'results_sha256': digest(directory/'results.h5'), 'config_sha256': digest(directory/'config.json'),
        'render_source_sha256': digest(Path(__file__)), 'frames': len(data.times),
        'options': {'fps': fps, 'particles': particles, 'seed': seed, 'dpi': dpi},
        'actual_times': data.times.tolist(), 'requested_times': data.requested_times.tolist(),
        'physical_interval': [float(data.times[0]), float(data.times[-1])],
        'duration_seconds': len(data.times)/fps, 'reynolds': cfg['lid_speed']*cfg['lx']/cfg['nu'],
        'max_divergence': data.max_divergence, 'max_stage_residual': data.max_stage_residual,
        'tracer_semantics': 'RK2, bilinear MAC faces, linear saved-time interpolation; no dye PDE',
        'streamline_semantics': 'instantaneous contours of psi(x,y)=integral_0^y u(x,s) ds',
        'validation_limits': 'startup visualisation; no steady, grid/time convergence or Ghia-data claim'}
    write_json(report/'animation.json', summary)
    (report/'animation.log').write_text(f'Rendering {directory}\n')
    print(f'Report: {report}', flush=True)
    try:
        paths, clipped = tracer_paths(data, count=particles, seed=seed)
        np.save(report/'tracer_paths.npy', paths, allow_pickle=False)
        summary['tracer_boundary_clips'] = clipped
        uref = cfg['lid_speed']
        omega_limit = max(float(np.percentile(np.abs(data.vorticity)*grid.lx/uref, 98)), 1e-12)
        psi_min, psi_max = float(data.streamfunction.min()), float(data.streamfunction.max())
        negative = -np.geomspace(max(abs(psi_min)*.98, 1e-7), 1e-6, 14) if psi_min < -1e-6 else np.array([])
        positive = np.geomspace(1e-6, psi_max*.98, 4) if psi_max > 2e-6 else np.array([])
        levels = np.unique(np.r_[negative, positive])
        tex: Any = {'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}',
            'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 13,
            'axes.titlesize': 15, 'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
            'pgf.preamble': r'\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{amsmath}'}
        with mpl.rc_context(tex):
            figure = Figure(figsize=(13.6, 7.4), dpi=dpi, facecolor='#f8f9fb')
            canvas = FigureCanvasAgg(figure)
            gs = figure.add_gridspec(1, 2, left=.065, right=.925, bottom=.13, top=.76, wspace=.30)
            speed_ax, omega_ax = figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1])
            speed = speed_ax.imshow(np.zeros_like(data.speed[0]), origin='lower',
                extent=(0., grid.lx, 0., grid.ly), cmap='viridis', vmin=0., vmax=1., interpolation='bilinear')
            omega = omega_ax.imshow(np.zeros_like(data.vorticity[0]), origin='lower',
                extent=(.5*grid.hx, grid.lx-.5*grid.hx, .5*grid.hy, grid.ly-.5*grid.hy),
                cmap='RdBu_r', vmin=-omega_limit, vmax=omega_limit, interpolation='bilinear')
            speed_ax.set_title(r'Speed $|\boldsymbol{u}|/U$, streamlines and tracers', loc='left', pad=27)
            omega_ax.set_title(r'Interior vorticity $\omega L/U$', loc='left', pad=27)
            for axis, image in ((speed_ax, speed), (omega_ax, omega)):
                axis.set(xlim=(0., grid.lx), ylim=(0., grid.ly), xlabel='$x$', ylabel='$y$', aspect='equal')
                for spine in axis.spines.values():
                    spine.set_linewidth(2.)
                    spine.set_color('#172536')
                axis.annotate('', xy=(.82*grid.lx, 1.035*grid.ly), xytext=(.18*grid.lx, 1.035*grid.ly),
                    arrowprops={'arrowstyle': '->', 'color': '#c24b2d', 'lw': 3}, annotation_clip=False)
                position = axis.get_position()
                cax = figure.add_axes((position.x1+.012, position.y0, .012, position.height))
                figure.colorbar(image, cax=cax, extend='both' if axis is omega_ax else 'neither')
            figure.text(.065, .94, 'Lid-driven cavity: startup from rest', fontsize=25,
                        weight='bold', color='#172536')
            figure.text(.065, .89, rf'$Re={summary["reynolds"]:g}$  $\cdot$  $U={uref:g}$  '
                        +rf'$\cdot$  $\nu={cfg["nu"]:g}$  $\cdot$  MAC ${grid.nx}\times{grid.ny}$',
                        fontsize=14, color='#526275')
            figure.text(.065, .855, 'Top lid moves right; side and bottom walls are stationary.  '
                        +cfg['scheme'].replace('_', '-'), fontsize=12, color='#526275')
            figure.text(.065, .047, 'Saved simulation fields; fixed colour scales. White particles show reconstructed trajectories.',
                        fontsize=11, color='#526275')
            speed.set_visible(False)
            omega.set_visible(False)
            figure.savefig(report/'layout.pdf', backend='pgf')
            figure.savefig(report/'layout.png', backend='pgf', dpi=dpi)
            font_probe = subprocess.run(['pdffonts', str(report/'layout.pdf')],
                                        check=True, capture_output=True, text=True)
            (report/'fonts.txt').write_text(font_probe.stdout)
            if 'TimesNewRoman' not in font_probe.stdout.replace('-', '').replace(' ', ''):
                raise RuntimeError('Times New Roman font verification failed')
            background = Image.open(report/'layout.png').convert('RGBA')
            figure.patch.set_alpha(0.)
            for axis in figure.axes:
                axis.patch.set_alpha(0.)
                axis.xaxis.set_visible(False)
                axis.yaxis.set_visible(False)
                for location in ('left', 'center', 'right'):
                    axis.set_title('', loc=location)
                for spine in axis.spines.values(): spine.set_visible(False)
                for artist in (*axis.lines, *axis.collections, *axis.texts): artist.set_visible(False)
            for text in figure.texts: text.set_visible(False)
            speed.set_visible(True)
            omega.set_visible(True)
            markers = speed_ax.scatter([], [], s=5., color='white', alpha=.9, linewidths=0)
            tails = LineCollection([], colors='white', linewidths=.65, alpha=.4)
            speed_ax.add_collection(tails)
            contours: Any = None
            font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),
                                              fallback_to_default=False)
            clock_font = ImageFont.truetype(font_path, int(dpi*.23))
            width, height = background.size
            if width % 2 or height % 2:
                raise ValueError('Use an even pixel canvas for H.264 export')
            command = [ffmpeg, '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', '-', '-an',
                '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(report/'cavity_flow.mp4')]
            with (report/'ffmpeg.log').open('w') as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
                assert process.stdin is not None
                try:
                    for index, time in enumerate(data.times):
                        speed.set_data(data.speed[index]/uref)
                        omega.set_data(data.vorticity[index]*grid.lx/uref)
                        if contours is not None: contours.remove()
                        if len(levels) and np.ptp(data.streamfunction[index]) > 1e-8:
                            contours = speed_ax.contour(np.arange(grid.nx+1)*grid.hx,
                                np.arange(grid.ny+1)*grid.hy, data.streamfunction[index],
                                levels=levels, colors='white', linewidths=.55, alpha=.45, linestyles='solid')
                        else:
                            contours = None
                        markers.set_offsets(paths[index])
                        history = paths[max(0, index-6):index+1]
                        segments = np.stack([history[:-1], history[1:]], axis=2).reshape(-1, 2, 2)
                        tails.set_segments(list(segments))
                        canvas.draw()
                        frame = Image.alpha_composite(background,
                            Image.fromarray(np.asarray(canvas.buffer_rgba()))).convert('RGB')
                        ImageDraw.Draw(frame).text((int(.94*width), int(.067*height)),
                            f't = {time:5.2f}', font=clock_font, fill='#172536', anchor='ra')
                        process.stdin.write(frame.tobytes())
                        if index in {0, len(data.times)//4, len(data.times)//2,
                                     3*len(data.times)//4, len(data.times)-1}:
                            frame.save(report/f'frame_{index:04d}.png')
                        if index % 50 == 0:
                            message = f'frame={index}/{len(data.times)} t={time:.8g}'
                            print(message, flush=True)
                            with (report/'animation.log').open('a') as log: log.write(message+'\n')
                finally:
                    process.stdin.close()
                    returncode = process.wait()
                if returncode:
                    raise RuntimeError(f'ffmpeg failed; see {report / "ffmpeg.log"}')
            subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', str(report/'cavity_flow.mp4'),
                '-filter_complex', '[0:v]fps=15,scale=960:-1:flags=lanczos,split[a][b];'
                '[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a',
                '-loop', '0', str(report/'cavity_flow.gif')], check=True)
            summary.update(status='complete', font_verified=True, clock_font=font_path,
                width=width, height=height, speed_colour_limits=[0., 1.],
                vorticity_colour_limits=[-omega_limit, omega_limit],
                vorticity_clipping_percentile=98, outputs={name: digest(report/name) for name in
                    ('cavity_flow.mp4', 'cavity_flow.gif', 'layout.pdf', 'layout.png', 'tracer_paths.npy')})
    except Exception as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(report/'animation.json', summary)
        with (report/'animation.log').open('a') as log:
            log.write(summary['status']+'\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--particles', type=int, default=500)
    parser.add_argument('--dpi', type=int, default=100)
    args = parser.parse_args()
    print(render(args.run, root=args.root, fps=args.fps, particles=args.particles, dpi=args.dpi))


if __name__ == '__main__':
    main()

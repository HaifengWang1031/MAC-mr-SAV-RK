"""Render saved periodic velocity fields as a Kolmogorov-flow movie.

The PDE is never advanced here. Fourier differentiation supplies vorticity and
the instantaneous streamfunction; periodic bilinear interpolation and RK2 supply
illustrative particle trajectories. Colour scales remain fixed throughout.
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

from experiments.workflow import PROJECT, digest, write_json
from solver.fourier import FourierSpace
from solver.fourier.space import RealArray
from .run import load_periodic


@dataclass
class MovieData:
    config: dict[str, Any]
    times: RealArray
    requested_times: RealArray
    u: RealArray
    v: RealArray
    speed: RealArray
    vorticity: RealArray
    streamfunction: RealArray
    mean_velocity: RealArray
    max_divergence: float
    max_stage_residual: float


def read_movie(directory: Path) -> MovieData:
    """Validate an explicit complete periodic record and derive snapshot observables."""
    cfg, _ = load_periodic(directory, require_complete=True)
    space = FourierSpace(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    with h5py.File(directory/'results.h5', 'r') as saved:
        times, requested = saved['snapshots/actual_times'][:], saved['snapshots/requested_times'][:]
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Animation needs distinct increasing snapshot times')
        if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-9, atol=1e-12):
            raise ValueError('Uniform actual snapshot spacing is required')
        u, v = saved['snapshots/u'][:], saved['snapshots/v'][:]
        residual = float(np.max(saved['stages/residual'][:], initial=0.))
        divergence = float(np.max(saved['diagnostics/divergence_inf'][:], initial=0.))
    if u.shape != (len(times), *space.shape) or v.shape != u.shape \
            or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError('Invalid saved periodic fields')
    velocity = space.to_spectral(np.stack([u, v], axis=1), filter=False)
    if np.max(np.abs(velocity[..., ~space.mask])) > 1e-11*(1+np.max(np.abs(velocity))):
        raise ValueError('Saved fields violate the retained Fourier band')
    reconstructed_divergence = float(np.max(np.abs(space.to_physical(space.divergence(velocity)))))
    if reconstructed_divergence > 100*cfg['stokes_tolerance']:
        raise ValueError('Saved-field full divergence check failed')
    curl = 1j*(space.kx*velocity[:, 1]-space.ky*velocity[:, 0])
    return MovieData(cfg, times, requested, u, v, np.hypot(u, v), space.to_physical(curl),
        space.to_physical(curl*space.inverse_k2), np.stack([u.mean(axis=(1, 2)), v.mean(axis=(1, 2))], axis=1),
        max(divergence, reconstructed_divergence), residual)


def sample_velocity(u: RealArray, v: RealArray, points: RealArray, lx: float, ly: float) -> RealArray:
    """Bilinear interpolation on endpoint-free collocation nodes, periodic in x and y."""
    ny, nx = u.shape
    x, y = (points[:, 0] % lx)*nx/lx, (points[:, 1] % ly)*ny/ly
    i, j = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x-i, y-j
    a, b, c, d = i % nx, (i+1) % nx, j % ny, (j+1) % ny
    columns = []
    for values in (u, v):
        bottom = (1-fx)*values[c, a]+fx*values[c, b]
        top = (1-fx)*values[d, a]+fx*values[d, b]
        columns.append((1-fy)*bottom+fy*top)
    return np.column_stack(columns)


def tracer_paths(data: MovieData, *, count: int = 500, seed: int = 20261019) -> RealArray:
    """Periodic midpoint trajectories in linearly time-interpolated saved fields."""
    if type(count) is not int or count < 1:
        raise ValueError('Tracer count must be a positive integer')
    lx, ly = data.config['lx'], data.config['ly']
    lengths = np.array([lx, ly])
    positions = np.random.default_rng(seed).uniform(0., 1., (count, 2))*lengths
    paths = np.empty((len(data.times), count, 2))
    paths[0] = positions
    for index, interval in enumerate(np.diff(data.times)):
        subdivisions = max(1, int(np.ceil(interval/.005)))
        step = interval/subdivisions
        for j in range(subdivisions):
            fraction = j/subdivisions
            u = (1-fraction)*data.u[index]+fraction*data.u[index+1]
            v = (1-fraction)*data.v[index]+fraction*data.v[index+1]
            midpoint = positions+.5*step*sample_velocity(u, v, positions, lx, ly)
            fraction = (j+.5)/subdivisions
            u = (1-fraction)*data.u[index]+fraction*data.u[index+1]
            v = (1-fraction)*data.v[index]+fraction*data.v[index+1]
            positions = (positions+step*sample_velocity(u, v, midpoint, lx, ly)) % lengths
        paths[index+1] = positions
    return paths


def render(directory: Path, *, root: Path = PROJECT, fps: int = 20,
           particles: int = 500, seed: int = 20261019, dpi: int = 100) -> Path:
    """Export an H.264 movie, GIF, preview frames and auditable rendering metadata."""
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
    lx, ly = cfg['lx'], cfg['ly']
    hx, hy = lx/cfg['nx'], ly/cfg['ny']
    report = root/'reports'/'periodic_animation'/(directory.name+'-'+
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:6])
    report.mkdir(parents=True)
    summary: dict[str, Any] = {'status': 'rendering', 'run': str(directory), 'run_id': directory.name,
        'results_sha256': digest(directory/'results.h5'), 'config_sha256': digest(directory/'config.json'),
        'render_source_sha256': digest(Path(__file__)), 'frames': len(data.times),
        'options': {'fps': fps, 'particles': particles, 'seed': seed, 'dpi': dpi},
        'actual_times': data.times.tolist(), 'requested_times': data.requested_times.tolist(),
        'duration_seconds': len(data.times)/fps, 'max_divergence': data.max_divergence,
        'max_stage_residual': data.max_stage_residual, 'coordinate_frame': 'laboratory',
        'tracer_semantics': 'periodic bilinear collocated interpolation; RK2 dt<=0.005; linear saved-time interpolation; no dye PDE',
        'streamline_semantics': 'Fourier psi=curl/k^2 plus mean_u*y-mean_v*x',
        'validation_limits': 'illustrative saved-field evolution; no turbulence or grid/time convergence claim'}
    write_json(report/'animation.json', summary)
    (report/'animation.log').write_text(f'Rendering {directory}\n')
    print(f'Report: {report}', flush=True)
    try:
        paths = tracer_paths(data, count=particles, seed=seed)
        np.save(report/'tracer_paths.npy', paths, allow_pickle=False)
        vmax = max(float(np.max(data.speed)), 1e-12)
        wmax = max(float(np.percentile(np.abs(data.vorticity), 99)), 1e-12)
        span = max(float(np.max(np.abs(data.streamfunction))), 1e-12)
        levels = np.linspace(-span, span, 19)
        tex: Any = {'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}',
            'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 13,
            'axes.titlesize': 15, 'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
            'pgf.preamble': r'\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{amsmath}'}
        with mpl.rc_context(tex):
            figure = Figure(figsize=(13.6, 7.4), dpi=dpi, facecolor='#f8f9fb')
            canvas = FigureCanvasAgg(figure)
            gs = figure.add_gridspec(1, 2, left=.065, right=.925, bottom=.13, top=.76, wspace=.30)
            speed_ax, omega_ax = figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1])
            extent = (-.5*hx, lx+.5*hx, -.5*hy, ly+.5*hy)
            speed = speed_ax.imshow(np.zeros((cfg['ny']+1, cfg['nx']+1)), origin='lower',
                extent=extent, cmap='viridis', vmin=0., vmax=vmax, interpolation='bilinear')
            omega = omega_ax.imshow(np.zeros((cfg['ny']+1, cfg['nx']+1)), origin='lower',
                extent=extent, cmap='RdBu_r', vmin=-wmax, vmax=wmax, interpolation='bilinear')
            speed_ax.set_title(r'Speed $|\boldsymbol{u}|$ and advected tracers', loc='left', pad=15)
            omega_ax.set_title(r'Vorticity $\omega$ and instantaneous streamlines', loc='left', pad=15)
            for axis, image in ((speed_ax, speed), (omega_ax, omega)):
                axis.set(xlim=(0., lx), ylim=(0., ly), xlabel='$x$', ylabel='$y$', aspect='equal')
                position = axis.get_position()
                cax = figure.add_axes((position.x1+.012, position.y0, .012, position.height))
                figure.colorbar(image, cax=cax, extend='both' if axis is omega_ax else 'neither')
            title = 'Kolmogorov flow: periodically forced vortices' if cfg['case'] == 'kolmogorov' else 'Periodic Navier--Stokes flow'
            figure.text(.065, .94, title, fontsize=25, weight='bold', color='#172536')
            force_label = (rf'$f_x=-{cfg["amplitude"]:g}\sin(2\pi {cfg["k_f"]}y/L_y)$  $\cdot$  '
                           if cfg['case'] == 'kolmogorov' else '')
            figure.text(.065, .89, force_label+rf'$\nu={cfg["nu"]:g}$  $\cdot$  '
                +rf'Fourier ${cfg["nx"]}\times{cfg["ny"]}$', fontsize=14, color='#526275')
            figure.text(.065, .85, r'Periodic $x$ and $y$  $\cdot$  '+cfg['scheme'].replace('_', '-')
                +r'  $\cdot$  strict $2/3$ filtering', fontsize=12, color='#526275')
            figure.text(.065, .047, 'Saved simulation fields; fixed colour scales. Tracers wrap across both periodic boundaries.',
                        fontsize=11, color='#526275')
            speed.set_visible(False)
            omega.set_visible(False)
            figure.savefig(report/'layout.pdf', backend='pgf')
            figure.savefig(report/'layout.png', backend='pgf', dpi=dpi)
            probe = subprocess.run(['pdffonts', str(report/'layout.pdf')], check=True, capture_output=True, text=True)
            (report/'fonts.txt').write_text(probe.stdout)
            if 'TimesNewRoman' not in probe.stdout.replace('-', '').replace(' ', ''):
                raise RuntimeError('Times New Roman font verification failed')
            background = Image.open(report/'layout.png').convert('RGBA')
            figure.patch.set_alpha(0.)
            for axis in figure.axes:
                axis.patch.set_alpha(0.)
                axis.xaxis.set_visible(False)
                axis.yaxis.set_visible(False)
                for location in ('left', 'center', 'right'): axis.set_title('', loc=location)
                for spine in axis.spines.values(): spine.set_visible(False)
                for artist in (*axis.lines, *axis.collections, *axis.texts): artist.set_visible(False)
            for text in figure.texts: text.set_visible(False)
            speed.set_visible(True)
            omega.set_visible(True)
            markers = speed_ax.scatter([], [], s=5., color='white', alpha=.9, linewidths=0)
            tails = LineCollection([], colors='white', linewidths=.65, alpha=.4)
            speed_ax.add_collection(tails)
            contours: Any = None
            font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
            clock_font = ImageFont.truetype(font_path, int(dpi*.23))
            width, height = background.size
            if width % 2 or height % 2: raise ValueError('H.264 needs an even pixel canvas')
            command = [ffmpeg, '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', '-', '-an',
                '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(report/'kolmogorov_flow.mp4')]
            with (report/'ffmpeg.log').open('w') as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
                assert process.stdin is not None
                try:
                    for index, time in enumerate(data.times):
                        speed.set_data(np.pad(data.speed[index], ((0, 1), (0, 1)), mode='wrap'))
                        omega.set_data(np.pad(data.vorticity[index], ((0, 1), (0, 1)), mode='wrap'))
                        if contours is not None: contours.remove()
                        psi = np.pad(data.streamfunction[index], ((0, 1), (0, 1)), mode='wrap')
                        xs, ys = np.arange(cfg['nx']+1)*hx, np.arange(cfg['ny']+1)*hy
                        psi += data.mean_velocity[index, 0]*ys[:, None]-data.mean_velocity[index, 1]*xs
                        contours = omega_ax.contour(xs, ys, psi, levels=levels, colors='#253a50',
                            linewidths=.5, alpha=.45) if np.ptp(psi) > 1e-10 else None
                        markers.set_offsets(paths[index])
                        history = paths[max(0, index-5):index+1]
                        segments = np.stack([history[:-1], history[1:]], axis=2).reshape(-1, 2, 2)
                        keep = np.all(np.abs(segments[:, 1]-segments[:, 0]) < np.array([lx, ly])/2, axis=1)
                        tails.set_segments(list(segments[keep]))
                        canvas.draw()
                        frame = Image.alpha_composite(background,
                            Image.fromarray(np.asarray(canvas.buffer_rgba()))).convert('RGB')
                        ImageDraw.Draw(frame).text((int(.94*width), int(.067*height)), f't = {time:5.2f}',
                            font=clock_font, fill='#172536', anchor='ra')
                        process.stdin.write(frame.tobytes())
                        if index in {0, len(data.times)//4, len(data.times)//2, 3*len(data.times)//4, len(data.times)-1}:
                            frame.save(report/f'frame_{index:04d}.png')
                        if index % 50 == 0:
                            message = f'frame={index}/{len(data.times)} t={time:.8g}'
                            print(message, flush=True)
                            with (report/'animation.log').open('a') as log: log.write(message+'\n')
                finally:
                    process.stdin.close()
                    returncode = process.wait()
                if returncode: raise RuntimeError(f'ffmpeg failed; see {report / "ffmpeg.log"}')
            subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', str(report/'kolmogorov_flow.mp4'),
                '-filter_complex', '[0:v]fps=15,scale=960:-1:flags=lanczos,split[a][b];'
                '[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a',
                '-loop', '0', str(report/'kolmogorov_flow.gif')], check=True)
            summary.update(status='complete', font_verified=True, clock_font=font_path, width=width, height=height,
                speed_colour_limits=[0., vmax], vorticity_colour_limits=[-wmax, wmax],
                vorticity_clipping_percentile=99, outputs={name: digest(report/name) for name in
                    ('kolmogorov_flow.mp4', 'kolmogorov_flow.gif', 'layout.pdf', 'layout.png', 'tracer_paths.npy')})
    except Exception as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(report/'animation.json', summary)
        with (report/'animation.log').open('a') as log: log.write(summary['status']+'\n')
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

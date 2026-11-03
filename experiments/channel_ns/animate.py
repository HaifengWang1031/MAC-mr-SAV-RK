"""Render an explicit saved channel run; this module never advances the PDE.

Purpose: visualise pressure-driven flow and perturbation relaxation with fixed
colour scales. Tracers are postprocessed from saved MAC faces, not a dye equation.
Static labels use XeLaTeX/Times New Roman; transparent Agg data layers make movie
export inexpensive without changing the saved numerical fields.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.records import PROJECT, digest, write_json
from solver.channel import ChannelGrid
from solver.mac.grid import Array
from .problems import poiseuille
from .records import load_channel


@dataclass
class MovieData:
    config: dict[str, Any]
    times: Array
    requested_times: Array
    u: Array
    v: Array
    speed: Array
    perturbation_vorticity: Array
    profiles: Array
    perturbation_norm: Array
    max_divergence: float
    max_stage_residual: float
    initial_mode: dict[str, Any] | None = None


def read_movie(directory: Path) -> MovieData:
    """Validate the public record and derive observables at actual saved times."""
    cfg, _ = load_channel(directory, require_complete=True)
    grid = ChannelGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    with h5py.File(directory/'results.h5', 'r') as data:
        times = data['snapshots/actual_times'][:]
        requested = data['snapshots/requested_times'][:]
        u, v = data['snapshots/u'][:], data['snapshots/v'][:]
        maximum_divergence = float(np.max(data['diagnostics/divergence_inf'][:]))
        maximum_residual = float(np.max(data['stages/residual'][:], initial=0.))
        mode = dict(data['initial_mode'].attrs) if 'initial_mode' in data else None
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Animation needs at least two distinct, increasing snapshot times')
    if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-9, atol=1e-12):
        raise ValueError('Use uniformly spaced actual snapshots for constant-speed playback')
    if u.shape != (len(times), grid.ny, grid.nx+1) or v.shape != (len(times), grid.ny+1, grid.nx):
        raise ValueError('Invalid channel snapshot shapes')
    uc, vc = .5*(u[:, :, :-1]+u[:, :, 1:]), .5*(v[:, :-1]+v[:, 1:])
    base_u, _ = grid.unpack(poiseuille(grid, cfg['nu'], cfg['driving'], discrete=True))
    omega = (v[:, 1:-1]-np.roll(v[:, 1:-1], 1, axis=2))/grid.hx \
        -np.diff(u[:, :, :-1], axis=1)/grid.hy
    omega_base = -np.diff(base_u[:, :-1], axis=0)/grid.hy
    norm = np.sqrt(grid.area*(np.sum((u[:, :, :-1]-base_u[:, :-1])**2, axis=(1, 2))
                              +np.sum(v[:, 1:-1]**2, axis=(1, 2))))
    return MovieData(cfg, times, requested, u, v, np.hypot(uc, vc), omega-omega_base,
                     np.mean(u[:, :, :-1], axis=2), norm, maximum_divergence, maximum_residual, mode)


def periodic_shift(field: Array, displacement_cells: float) -> Array:
    """Evaluate a real periodic row field at x plus displacement; display only."""
    nx = field.shape[1]
    frequencies = 2*np.pi*np.fft.rfftfreq(nx)
    return np.fft.irfft(np.fft.rfft(field, axis=1)*np.exp(1j*frequencies*displacement_cells),
                       n=nx, axis=1)


def sample_velocity(grid: ChannelGrid, u: Array, v: Array, points: Array) -> Array:
    """Bilinear MAC-face interpolation, periodic x and zero velocity at both walls."""
    result = np.empty_like(points)
    y = np.clip(points[:, 1], 0., grid.ly)
    for component, faces, offset, nodes in (
        (0, np.vstack([np.zeros((1, grid.nx)), u[:, :-1], np.zeros((1, grid.nx))]),
         0., np.r_[0., (np.arange(grid.ny)+.5)*grid.hy, grid.ly]),
        (1, v, .5, np.arange(grid.ny+1)*grid.hy),
    ):
        xindex = (points[:, 0] % grid.lx)/grid.hx-offset
        left = np.floor(xindex).astype(int)
        fraction_x = xindex-left
        lower = np.clip(np.searchsorted(nodes, y, side='right')-1, 0, len(nodes)-2)
        fraction_y = (y-nodes[lower])/(nodes[lower+1]-nodes[lower])
        a, b = left % grid.nx, (left+1) % grid.nx
        bottom = (1-fraction_x)*faces[lower, a]+fraction_x*faces[lower, b]
        top = (1-fraction_x)*faces[lower+1, a]+fraction_x*faces[lower+1, b]
        result[:, component] = (1-fraction_y)*bottom+fraction_y*top
    return result


def tracer_paths(data: MovieData, *, count: int = 420, seed: int = 20261005) -> Array:
    """RK2 paths in a linearly time-interpolated saved velocity field."""
    if count < 1:
        raise ValueError('Tracer count must be positive')
    cfg = data.config
    grid = ChannelGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    rng = np.random.default_rng(seed)
    positions = np.column_stack([rng.uniform(0., grid.lx, count),
                                 rng.uniform(.04*grid.ly, .96*grid.ly, count)])
    paths = np.empty((len(data.times), count, 2))
    paths[0] = positions
    for index, interval in enumerate(np.diff(data.times)):
        subdivisions = max(1, int(np.ceil(interval/.01)))
        step = interval/subdivisions
        for j in range(subdivisions):
            fraction = j/subdivisions
            u = (1-fraction)*data.u[index]+fraction*data.u[index+1]
            v = (1-fraction)*data.v[index]+fraction*data.v[index+1]
            midpoint = positions+.5*step*sample_velocity(grid, u, v, positions)
            fraction = (j+.5)/subdivisions
            u = (1-fraction)*data.u[index]+fraction*data.u[index+1]
            v = (1-fraction)*data.v[index]+fraction*data.v[index+1]
            positions += step*sample_velocity(grid, u, v, midpoint)
            positions[:, 0] %= grid.lx
            positions[:, 1] = np.clip(positions[:, 1], 0., grid.ly)
        paths[index+1] = positions
    return paths


def render(directory: Path, *, root: Path = PROJECT, fps: int = 20,
           particles: int = 420, seed: int = 20261005, dpi: int = 120,
           near_wall: bool = False, output_directory: Path | None = None) -> Path:
    """Export MP4, GIF and PNG previews from a verified complete run only."""
    if fps < 1 or dpi < 60:
        raise ValueError('Require positive fps and dpi >= 60')
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None or shutil.which('xelatex') is None:
        raise RuntimeError('Animation export requires ffmpeg and XeLaTeX')
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib import font_manager

    directory = directory.resolve()
    data = read_movie(directory)
    cfg = data.config
    grid = ChannelGrid(cfg['nx'], cfg['ny'], cfg['lx'], cfg['ly'])
    ts = data.initial_mode
    if near_wall and ts is None:
        raise ValueError('Near-wall instability view requires a saved TS-mode run')
    paths = tracer_paths(data, count=particles, seed=seed)
    phase_speed = 0. if ts is None else -float(ts['eigenvalue_imag'])*grid.lx/(2*np.pi)
    if ts is not None:
        paths[:, :, 0] = (paths[:, :, 0]-phase_speed*(data.times-data.times[0])[:, None]) % grid.lx
    report = (root/'reports'/'channel_animation'/directory.name if output_directory is None
              else output_directory.resolve())
    report.mkdir(parents=True, exist_ok=True)
    options = {'fps': fps, 'particles': particles, 'seed': seed, 'dpi': dpi, 'near_wall': near_wall}
    summary: dict[str, Any] = {
        'status': 'rendering', 'run': str(directory), 'run_id': directory.name,
        'results_sha256': digest(directory/'results.h5'),
        'render_source_sha256': digest(Path(__file__)), 'options': options,
        'frames': len(data.times), 'actual_times': data.times.tolist(),
        'requested_times': data.requested_times.tolist(),
        'duration_seconds': len(data.times)/fps,
        'max_divergence': data.max_divergence, 'max_stage_residual': data.max_stage_residual,
        'perturbation_L2_initial': float(data.perturbation_norm[0]),
        'perturbation_L2_final': float(data.perturbation_norm[-1]),
        'tracer_semantics': 'RK2, bilinear MAC faces, linear saved-time interpolation; no dye PDE',
        'validation_limits': 'Illustrative movie; nonlinear endpoint convergence is not certified',
        'coordinate_frame': 'laboratory' if ts is None else 'x minus initial TS phase speed times t',
        'phase_speed': phase_speed,
        'vorticity_y_limits': [0., .2*grid.ly if near_wall else grid.ly],
        'vertical_zoom': 5. if near_wall else 1.,
        'vorticity_clip_percentile': 99 if near_wall else None,
        'vorticity_absolute_max': float(np.max(np.abs(data.perturbation_vorticity))),
        'streamline_level_rule': ('14 levels through the instantaneous negative near-wall streamfunction plus positive levels'
                                  if near_wall else 'existing full-height wave-frame contour rule'),
    }
    if ts is not None:
        summary.update(reynolds_half_height=float(ts['reynolds_half_height']),
                       initial_peak_perturbation_fraction=cfg['epsilon'],
                       perturbation_amplitude_gain=float(data.perturbation_norm[-1]/data.perturbation_norm[0]))
    write_json(report/'animation.json', summary)
    (report/'animation.log').write_text(f'Rendering {directory}\n{json.dumps(options)}\n')
    tex: Any = {'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}',
           'font.family': 'serif', 'font.serif': ['Times New Roman'],
           'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13,
           'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
           'pgf.preamble': r'\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{amsmath}'}
    with mpl.rc_context(tex):
        figure = Figure(figsize=(12, 7.5), dpi=dpi, facecolor='#f8f9fb')
        canvas = FigureCanvasAgg(figure)
        gs = figure.add_gridspec(2, 2, left=.065, right=.975, bottom=.11, top=.79,
                                width_ratios=[3.5, 1.], hspace=.46, wspace=.32)
        speed_ax, profile_ax = figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1])
        omega_ax, norm_ax = figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1])
        flow_axes = (speed_ax, omega_ax)
        vmax = max(float(np.max(data.speed)), 1e-12)
        wmax = max(float(np.max(np.abs(data.perturbation_vorticity))), 1e-12)
        if near_wall:
            vertices_y = np.arange(1, grid.ny)*grid.hy
            wmax = max(float(np.percentile(np.abs(
                data.perturbation_vorticity[:, vertices_y <= .2*grid.ly]), 99)), 1e-12)
        speed = speed_ax.imshow(np.zeros_like(data.speed[0]), origin='lower',
                               extent=(0., grid.lx, 0., grid.ly), cmap='viridis',
                               vmin=0., vmax=vmax, aspect='auto', interpolation='bilinear')
        omega = omega_ax.imshow(np.zeros((grid.ny-1, grid.nx+1)), origin='lower',
                               extent=(-.5*grid.hx, grid.lx+.5*grid.hx,
                                       .5*grid.hy, grid.ly-.5*grid.hy),
                               cmap='RdBu_r', vmin=-wmax, vmax=wmax,
                               aspect='auto', interpolation='bilinear')
        for axis in flow_axes:
            axis.set(xlim=(0., grid.lx), ylim=(0., grid.ly), xlabel='$x$', ylabel='$y$')
            axis.axhline(0., color='#172536', lw=3., clip_on=False)
            axis.axhline(grid.ly, color='#172536', lw=3., clip_on=False)
        speed_ax.set_title(r'Speed $|\boldsymbol{u}|$ and advected tracers', loc='left', pad=13)
        omega_ax.set_title(r'Perturbation vorticity $\omega-\omega_P$', loc='left', pad=13)
        if ts is not None:
            for axis in flow_axes: axis.set_xlabel('$x-ct$')
            speed_ax.set_title('Speed and tracers in wave coordinates', loc='left', pad=13)
            omega_ax.set_title('Perturbation vorticity and wave-frame streamlines', loc='left', pad=13)
        if near_wall:
            omega_ax.set_ylim(0., .2*grid.ly)
            omega_ax.set_title('Lower-wall zoom: perturbation vorticity and closed streamlines', loc='left', pad=13)
        for axis, image in ((speed_ax, speed), (omega_ax, omega)):
            position = axis.get_position()
            cax = figure.add_axes((position.x1+.01, position.y0, .011, position.height))
            figure.colorbar(image, cax=cax, extend='both' if near_wall and axis is omega_ax else 'neither')
        yc = (np.arange(grid.ny)+.5)*grid.hy
        baseline = cfg['driving']*yc*(grid.ly-yc)/(2*cfg['nu'])
        profile_ax.plot(baseline, yc, '--', color='#8895a5', lw=1.8, label='Poiseuille')
        profile_ax.set(xlabel=r'$\overline{u}(y)$', ylabel='$y$', ylim=(0., grid.ly),
                       xlim=(min(0., float(data.profiles.min())*1.1),
                             max(float(data.profiles.max()), float(baseline.max()), 1e-6)*1.15))
        profile_ax.set_title('Mean profile', loc='left', pad=13)
        profile_ax.legend(loc='lower right', fontsize=10, frameon=False)
        lower = max(float(data.perturbation_norm.min())*.7, 1e-12)
        upper = max(float(data.perturbation_norm.max())*1.4, lower*10)
        norm_ax.set(yscale='log', xlabel='$t$', ylabel=r'$\|\boldsymbol{u}-\boldsymbol{u}_P\|_h$',
                    xlim=(data.times[0], data.times[-1]), ylim=(lower, upper))
        norm_ax.set_title('Disturbance amplitude', loc='left', pad=13)
        norm_ax.grid(alpha=.2)
        if ts is not None:
            prediction = data.perturbation_norm[0]*np.exp(float(ts['eigenvalue_real'])*(data.times-data.times[0]))
            norm_ax.set_ylim(lower, max(upper, float(prediction.max())*1.15))
            norm_ax.plot(data.times, prediction, '--', color='#8895a5', lw=1.2, label='Linear prediction')
            norm_ax.legend(loc='upper left', frameon=False, fontsize=10)
        figure.text(.065, .925, 'Pressure-driven channel flow' if ts is None else 'Poiseuille flow: TS wave growth',
                    fontsize=25, weight='bold', color='#172536')
        physical_label = (rf'$\nu={cfg["nu"]:g}$  $\cdot$  $F_x={cfg["driving"]:g}$' if ts is None else
                          rf'$Re={ts["reynolds_half_height"]:g}$  $\cdot$  $\alpha={ts["alpha_half_height"]:g}$')
        if near_wall:
            physical_label += rf'  $\cdot$  initial peak disturbance ${100*cfg["epsilon"]:g}\%\,U_{{max}}$'
        figure.text(.065, .872, r'Periodic $x$  $\cdot$  No-slip walls  $\cdot$  '+physical_label,
                    fontsize=14, color='#526275')
        figure.text(.065, .845, rf'MAC ${grid.nx}\times{grid.ny}$  $\cdot$  '
                    +rf'$L_x={grid.lx:g},\ H={grid.ly:g}$  $\cdot$  {cfg["scheme"].replace("_", "-")}',
                    fontsize=12, color='#526275')
        footer = ('5x lower-wall zoom; fixed 99th-percentile vorticity clip. Wave coordinates; streamlines relative to c.'
                  if near_wall else 'Saved simulation fields; fixed colour scales. Tracers are reconstructed trajectories.')
        figure.text(.065, .037, footer,
                    fontsize=11, color='#526275')
        # Render labels once with the exact system font through XeLaTeX.
        speed.set_visible(False)
        omega.set_visible(False)
        figure.savefig(report/'layout.pdf', backend='pgf')
        figure.savefig(report/'layout.png', backend='pgf', dpi=dpi)
        background = Image.open(report/'layout.png').convert('RGBA')
        # Agg now draws only changing data. All visible lettering is already in the PGF layer.
        figure.patch.set_alpha(0.)
        for axis in figure.axes:
            axis.patch.set_alpha(0.)
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            axis.title.set_visible(False)
            axis.set_title('', loc='left')
            axis.set_title('', loc='right')
            for spine in axis.spines.values():
                spine.set_visible(False)
            for line in axis.lines:
                line.set_visible(False)
            for collection in axis.collections:
                collection.set_visible(False)
            legend = axis.get_legend()
            if legend is not None:
                legend.set_visible(False)
        for text in figure.texts:
            text.set_visible(False)
        speed.set_visible(True)
        omega.set_visible(True)
        markers = speed_ax.scatter([], [], s=5., color='white', alpha=.85, linewidths=0)
        tails = LineCollection([], colors='white', linewidths=.6, alpha=.3)
        speed_ax.add_collection(tails)
        profile, = profile_ax.plot([], [], color='#157a8a', lw=2.5)
        disturbance, = norm_ax.plot([], [], color='#c45339', lw=2.2)
        norm_marker, = norm_ax.plot([], [], 'o', color='#c45339', ms=5)
        contours: Any = None
        font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),
                                          fallback_to_default=False)
        clock_font = ImageFont.truetype(font_path, int(dpi*.23))
        width, height = background.size
        command = [ffmpeg, '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                   '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', '-',
                   '-an', '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(report/'channel_flow.mp4')]
        with (report/'ffmpeg.log').open('w') as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            assert process.stdin is not None
            try:
                for index, time in enumerate(data.times):
                    displacement = phase_speed*(time-data.times[0])/grid.hx
                    speed.set_data(data.speed[index] if ts is None else periodic_shift(data.speed[index], displacement))
                    omega_field = data.perturbation_vorticity[index] if ts is None else \
                        periodic_shift(data.perturbation_vorticity[index], displacement)
                    omega.set_data(np.pad(omega_field, ((0, 0), (0, 1)), mode='wrap'))
                    if ts is not None:
                        u_wave = periodic_shift(data.u[index, :, :-1], displacement)
                        vertices_y = np.arange(grid.ny+1)*grid.hy
                        psi = np.vstack([np.zeros((1, grid.nx)), grid.hy*np.cumsum(u_wave, axis=0)]) \
                            -phase_speed*vertices_y[:, None]
                        mean = np.mean(psi, axis=1)
                        span = max(float(np.max(np.ptp(psi, axis=1)))*.7, 1e-4)
                        levels = np.unique(np.r_[np.linspace(mean.min(), mean.max(), 10),
                                                  mean.min()+np.linspace(-span, span, 11),
                                                  mean.max()+np.linspace(-span, span, 11)])
                        if near_wall:
                            local = psi[vertices_y <= .2*grid.ly]
                            minimum, maximum = float(local.min()), float(local.max())
                            negative = (np.linspace(.98*minimum, .02*minimum, 14)
                                        if minimum < -1e-10 else np.array([]))
                            positive = (np.geomspace(max(1e-9, .0001*maximum), .98*maximum, 8)
                                        if maximum > 1e-8 else np.array([]))
                            levels = np.unique(np.r_[negative, positive])
                        if contours is not None: contours.remove()
                        contours = omega_ax.contour(np.arange(grid.nx+1)*grid.hx, vertices_y,
                            np.column_stack([psi, psi[:, 0]]), levels=levels,
                            colors='#1b334e', linewidths=.55, alpha=.65)
                    markers.set_offsets(paths[index])
                    history = paths[max(0, index-7):index+1]
                    segments = np.stack([history[:-1], history[1:]], axis=2).reshape(-1, 2, 2)
                    if len(segments):
                        segments = segments[np.abs(np.diff(segments[:, :, 0], axis=1)[:, 0]) < grid.lx/2]
                    tails.set_segments(list(segments))
                    profile.set_data(data.profiles[index], yc)
                    disturbance.set_data(data.times[:index+1], data.perturbation_norm[:index+1])
                    norm_marker.set_data([time], [data.perturbation_norm[index]])
                    canvas.draw()
                    overlay = Image.fromarray(np.asarray(canvas.buffer_rgba()))
                    frame = Image.alpha_composite(background, overlay).convert('RGB')
                    ImageDraw.Draw(frame).text((int(.975*width), int(.052*height)),
                                               f't = {time:5.2f}', font=clock_font,
                                               fill='#172536', anchor='ra')
                    process.stdin.write(frame.tobytes())
                    if index in {0, len(data.times)//2, len(data.times)-1}:
                        frame.save(report/f'frame_{index:04d}.png')
                    if index % 50 == 0:
                        with (report/'animation.log').open('a') as stream:
                            stream.write(f'frame={index}/{len(data.times)} t={time:.8g}\n')
            finally:
                process.stdin.close()
                returncode = process.wait()
            if returncode:
                raise RuntimeError(f'ffmpeg failed; see {report / "ffmpeg.log"}')
        gif_command = [ffmpeg, '-y', '-v', 'error', '-i', str(report/'channel_flow.mp4'),
                       '-filter_complex', '[0:v]fps=15,scale=960:-1:flags=lanczos,split[a][b];'
                       '[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a',
                       '-loop', '0', str(report/'channel_flow.gif')]
        subprocess.run(gif_command, check=True)
        probe = subprocess.run(['pdffonts', str(report/'layout.pdf')], check=True,
                               capture_output=True, text=True) if shutil.which('pdffonts') else None
        if probe is not None:
            (report/'fonts.txt').write_text(probe.stdout)
            if 'TimesNewRoman' not in probe.stdout.replace('-', '').replace(' ', ''):
                raise RuntimeError('Could not verify Times New Roman in the exported label layer')
        summary.update(status='complete', clock_font=font_path, width=width, height=height,
                       speed_colour_limits=[0., vmax], perturbation_vorticity_colour_limits=[-wmax, wmax],
                       font_verified=probe is not None,
                       outputs={name: digest(report/name) for name in
                                ('channel_flow.mp4', 'channel_flow.gif', 'layout.pdf', 'layout.png')})
        write_json(report/'animation.json', summary)
        with (report/'animation.log').open('a') as stream:
            stream.write('complete\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--particles', type=int, default=420)
    parser.add_argument('--dpi', type=int, default=120)
    parser.add_argument('--near-wall', action='store_true', help='Show the lower 20 percent of a TS channel')
    parser.add_argument('--output', type=Path, help='Separate export directory; preserve existing movies')
    args = parser.parse_args()
    print(render(args.run, root=args.root, fps=args.fps, particles=args.particles, dpi=args.dpi,
                 near_wall=args.near_wall, output_directory=args.output))


if __name__ == '__main__':
    main()

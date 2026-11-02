"""Read explicit saved records and export fields, diagnostics and startup movies."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
from typing import Any

import h5py
import matplotlib as mpl
import matplotlib.tri as mtri
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.path import Path as PolygonPath
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import vstack
from scipy.spatial import cKDTree

from experiments.workflow import digest, write_json
from solver.fem.airfoil import airfoil_polygon
from .records import load


def analyze(directory: Path, *, output: Path, movie: bool = False,
            fps: int = 20, dpi: int = 130) -> dict[str, Any]:
    """No flow solve or mesh generation; saved coefficients are the only fields."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.is_relative_to(directory) or fps < 1 or dpi < 50:
        raise ValueError('Separate report directory, positive fps and dpi>=50 required')
    before = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    cfg, manifest, mesh, vb, pb = load(directory)
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(directory/'results.h5', 'r') as data:
        times = data['snapshots/actual_times'][:]
        u, v = data['snapshots/u'][:], data['snapshots/v'][:]
        pressures = data['snapshots/physical_pressure'][:]
        forces = data['snapshots/model_force_per_span'][:]
        diagnostics = {k: values[:] for k, values in data['diagnostics'].items()}
        nodes = data['times'][:]
        darcy_fluxes = json.loads(data['darcy'].attrs['fluxes']) if cfg['darcy_reference'] else None
    with (output/'diagnostics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t', *diagnostics])
        writer.writerows(zip(nodes, *diagnostics.values()))
    polygon = airfoil_polygon(cfg['chord'], cfg['angle'], cfg['surface_points'])
    scalar = vb.split_bases()[0]
    # Every P2 vertex/edge-midpoint value participates in this display subdivision.
    display = mesh.refined(1)
    distances, display_map = cKDTree(scalar.doflocs.T).query(display.p.T)
    assert np.max(distances) < 1e-11*cfg['chord']
    bounds = np.asarray(cfg['bounds'])*cfg['chord']
    views = [bounds, np.array([-.2, 1.2, -.4, .35])*cfg['chord']]
    probes, arrow_points = [], []
    for xmin, xmax, ymin, ymax in views:
        x, y = np.meshgrid(np.linspace(xmin+.03*(xmax-xmin), xmax-.03*(xmax-xmin), 26),
                           np.linspace(ymin+.03*(ymax-ymin), ymax-.03*(ymax-ymin), 14))
        points = np.array([x.ravel(), y.ravel()])
        points = points[:, ~PolygonPath(np.vstack((polygon, polygon[0]))).contains_points(points.T)]
        blocks = [scalar.probes(points[:, i:i+64]) for i in range(0, points.shape[1], 64)]
        probes.append(vstack(blocks).tocsr())
        arrow_points.append(points)
    speed_max = max(float(np.max(np.hypot(u, v))), 1e-14)
    settings: Any = {'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 11,
        'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
        'pgf.preamble': r'\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{amsmath}'}
    positions = [(.065, .21, .465, .55), (.63, .21, .27, .55)]
    size = (13.6, 5.1)
    with mpl.rc_context(settings):
        labels = Figure(figsize=size, dpi=dpi, facecolor='white')
        for i, (position, view) in enumerate(zip(positions, views)):
            axis = labels.add_axes(position)
            axis.set(xlim=view[:2], ylim=view[2:], xlabel='$x$', ylabel='$y$', aspect='equal',
                     title='Depth-averaged speed' if i == 0 else 'Airfoil: speed and velocity arrows')
            axis.add_patch(Polygon(polygon, fc='#d1dbe2', ec='#213747', lw=1.2))
            norm = mpl.colors.Normalize(0., speed_max)
            bar = labels.add_axes((position[0]+position[2]+.012, .26, .011, .40))
            labels.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=bar, label=r'$|\overline{\mathbf{u}}|$')
        labels.text(.065, .92, 'Hele--Shaw airfoil: inertial Brinkman model', fontsize=20)
        labels.text(.065, .84, rf'$\alpha={cfg["angle"]:g}^\circ$, $b/c={cfg["gap_ratio"]:g}$, '
                    rf'$\nu={cfg["nu"]:g}$, {cfg["scheme"].replace('_', '-')}', fontsize=12)
        labels.text(.065, .07, 'MWR depth-averaged model; fixed color scales; saved physical times.', fontsize=10)
        buffer = io.BytesIO()
        labels.savefig(buffer, format='png', backend='pgf', dpi=dpi)
    background = Image.open(buffer).convert('RGBA')
    dynamic = Figure(figsize=size, dpi=dpi, facecolor=(0, 0, 0, 0))
    canvas = FigureCanvasAgg(dynamic)
    images, arrows = [], []
    for i, (position, view) in enumerate(zip(positions, views)):
        axis = dynamic.add_axes(position)
        axis.set(xlim=view[:2], ylim=view[2:], aspect='equal')
        axis.axis('off')
        images.append(axis.tripcolor(mtri.Triangulation(display.p[0], display.p[1], display.t.T),
            np.zeros(display.p.shape[1]), shading='gouraud', cmap='viridis', vmin=0., vmax=speed_max))
        point = arrow_points[i]
        arrows.append(axis.quiver(point[0], point[1], np.zeros(point.shape[1]), np.zeros(point.shape[1]),
            color='white', edgecolor='#233747', linewidth=.20, width=.0025,
            scale_units='xy', scale=speed_max/((.20 if i == 0 else .045)*cfg['chord'])))
    width, height = canvas.get_width_height()
    if background.size != (width, height):
        background = background.resize((width, height), Image.Resampling.LANCZOS)
    font = ImageFont.truetype(font_manager.findfont('Times New Roman'), int(.17*dpi))

    def frame(index: int) -> Image.Image:
        speed = np.hypot(u[index], v[index])
        for i in range(2):
            images[i].set_array(speed[display_map])
            arrows[i].set_UVC(probes[i] @ u[index], probes[i] @ v[index])
        canvas.draw()
        composed = background.copy()
        composed.alpha_composite(Image.fromarray(np.asarray(canvas.buffer_rgba())))
        draw = ImageDraw.Draw(composed)
        draw.text((int(.72*width), int(.07*height)), f't = {times[index]:.5g}', fill='#233747', font=font)
        return composed.convert('RGB')

    frame(0).save(output/'initial.png')
    frame(len(times)-1).save(output/'final.png')
    if movie:
        video = output/'evolution.mp4'
        process = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
            '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p', '-crf', '20', str(video)], stdin=subprocess.PIPE)
        assert process.stdin is not None
        try:
            for i in range(len(times)):
                process.stdin.write(np.asarray(frame(i)).tobytes())
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Video encoder failed')
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
    assert before == {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    balance_power = (diagnostics['viscous_dissipation']+diagnostics['gap_dissipation']
                     +diagnostics['convective_boundary_power']-diagnostics['pressure_power'])
    balance_power += diagnostics.get('outlet_backflow_dissipation', 0.)
    balance_power += diagnostics.get('inlet_kinetic_energy_correction', 0.)
    balance = diagnostics['kinetic'][-1]-diagnostics['kinetic'][0]+float(np.trapezoid(balance_power, nodes))
    actual_re = abs(float(diagnostics['outlet_mean_velocity'][-1]))*cfg['chord']/cfg['nu']
    summary: dict[str, Any] = dict(status='complete', run_id=directory.name, run=str(directory),
        config=cfg, producer=manifest['source'], inputs_unchanged=True, input_fingerprints=before,
        pressure_constraint=cfg['pressure_constraint'],
        legacy_registry_pressure_label=cfg.get('hele_shaw_schema_version', 0) == 0,
        actual_times=times.tolist(), frames=len(times), temporal_interpolation=False,
        display='P2 nodal values on four linear subtriangles per saved element; exact P2 velocity arrows',
        final_time=float(nodes[-1]), max_abs_r=float(np.max(np.abs(diagnostics['r']))),
        final_pressure_range=[float(np.min(pressures[-1])), float(np.max(pressures[-1]))],
        final_model_force_per_span=forces[-1].tolist(), darcy_boundary_fluxes=darcy_fluxes,
        final_outlet_mean_velocity=float(diagnostics['outlet_mean_velocity'][-1]),
        actual_chord_reynolds=actual_re, actual_gap_reynolds=actual_re*cfg['gap_ratio'],
        reduced_inertia_parameter=actual_re*cfg['gap_ratio']**2,
        physical_energy_balance_trapezoid_diagnostic=balance,
        speed_limits=[0., speed_max], font='XeLaTeX Times New Roman; clock Times New Roman',
        limits='Reduced model only; no 3D validity, grid independence or aerodynamic force validation.')
    write_json(output/'analysis.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--movie', action='store_true')
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, output=args.output, movie=args.movie), indent=2))


if __name__ == '__main__':
    main()

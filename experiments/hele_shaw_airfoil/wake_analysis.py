"""Readonly wake diagnostics and signed-vorticity movies from explicit records.

Backflow, complex velocity-gradient eigenvalues and surface shear reversal are
reported separately. Vorticity by itself is not classified as a vortex. The
exclusion x/c>1.05 keeps the known sharp-tip singular region outside wake metrics.
"""
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
from scipy.sparse.linalg import splu
from skfem import BilinearForm, FacetBasis, asm

from experiments.records import digest, write_json
from solver.fem.airfoil import airfoil_polygon
from .records import load


def flow_indicators(velocity: np.ndarray, gradient: np.ndarray, weight: np.ndarray,
                    reference_speed: float, chord: float = 1.) -> dict[str, float]:
    """Physical quadrature statistics on a caller-selected fluid-only wake ROI."""
    if velocity.shape != (2, len(weight)) or gradient.shape != (2, 2, len(weight)) \
            or not len(weight) or np.any(weight <= 0.) or chord <= 0.:
        raise ValueError('Invalid wake quadrature arrays')
    if not all(np.isfinite(x).all() for x in (velocity, gradient, weight)):
        raise ValueError('Nonfinite wake field')
    trace = gradient[0, 0]+gradient[1, 1]
    determinant = gradient[0, 0]*gradient[1, 1]-gradient[0, 1]*gradient[1, 0]
    swirling = .5*np.sqrt(np.maximum(0., 4*determinant-trace*trace))
    omega = gradient[1, 0]-gradient[0, 1]
    scale = max(abs(reference_speed), 1e-9)
    backflow = velocity[0] < -1e-3*scale
    rotating = swirling > .05*scale/chord
    area = float(np.sum(weight))
    return dict(min_streamwise_velocity=float(velocity[0].min()),
        backflow_area=float(np.sum(weight[backflow])), backflow_fraction=float(np.sum(weight[backflow])/area),
        maximum_swirling_strength=float(swirling.max()),
        rotating_area=float(np.sum(weight[rotating])),
        enstrophy=float(np.sum(omega*omega*weight)),
        divergence_RMS=float(np.sqrt(np.sum(trace*trace*weight)/area)))


def analyze_wake(directory: Path, *, output: Path, movie: bool = True, fps: int = 20,
                 dpi: int = 120, vorticity_limit: float = 30.) -> dict[str, Any]:
    """Load only saved FE data; do not build a flow model, remesh or advance it."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.is_relative_to(directory) or fps < 1 or dpi < 50 or vorticity_limit <= 0.:
        raise ValueError('Invalid report location or rendering controls')
    before = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    cfg, manifest, mesh, vb, pb = load(directory)
    output.mkdir(parents=True, exist_ok=True)
    chord = cfg['chord']
    xmin, xmax, ymin, ymax = np.asarray(cfg['bounds'])*chord
    nominal_speed = cfg['gap']**2*cfg['pressure_drop']/(12*cfg['rho']*cfg['nu']*(xmax-xmin))
    points = np.asarray(vb.global_coordinates())
    roi = ((points[0] > 1.05*chord) & (points[0] < 5.*chord)
           & (points[1] > -.7*chord) & (points[1] < .5*chord))
    if not np.any(roi):
        raise ValueError('Saved mesh does not cover the wake ROI')
    foil_basis = FacetBasis(mesh, vb.elem, facets=mesh.boundaries['airfoil'],
                            intorder=cfg['quadrature_order'])
    surface = np.asarray(foil_basis.global_coordinates())
    angle = np.deg2rad(cfg['angle'])
    body_x = np.cos(angle)*(surface[0]-.25*chord)-np.sin(angle)*surface[1]+.25*chord
    body_y = np.sin(angle)*(surface[0]-.25*chord)+np.cos(angle)*surface[1]
    surface_region = (body_x > .2*chord) & (body_x < .97*chord)
    normal = -foil_basis.normals
    tangent = np.stack([normal[1], -normal[0]])
    tangent *= np.where(tangent[0] < 0., -1., 1.)[None]
    phi = np.stack([np.asarray(b[0]) for b in pb.basis])

    @BilinearForm
    def mass(p: Any, q: Any, w: Any) -> Any:
        return p*q

    projection = splu(asm(mass, pb).tocsc())
    polygon = airfoil_polygon(chord, cfg['angle'], cfg['surface_points'])
    camera = np.array([-.4, 5.2, -.8, .7])
    qx, qy = np.meshgrid(np.linspace(camera[0], camera[1], 42), np.linspace(camera[2], camera[3], 13))
    arrow_points = np.array([qx.ravel(), qy.ravel()])*chord
    mask = ~PolygonPath(np.vstack((polygon, polygon[0]))).contains_points(arrow_points.T)
    arrow_points = arrow_points[:, mask]
    scalar = vb.split_bases()[0]
    probe = vstack([scalar.probes(arrow_points[:, i:i+64]) for i in range(0, arrow_points.shape[1], 64)]).tocsr()
    with h5py.File(directory/'results.h5', 'r') as data:
        times = data['snapshots/actual_times'][:]
        nodes = data['times'][:]
        mean = data['diagnostics/outlet_mean_velocity'][:]
        scalar_r = data['diagnostics/r'][:]
        forces = data['snapshots/model_force_per_span'][:]

    settings: Any = {'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 11,
        'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
        'pgf.preamble': r'\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{amsmath}'}
    size, position = (13.2, 4.9), (.07, .18, .81, .60)
    with mpl.rc_context(settings):
        labels = Figure(figsize=size, dpi=dpi, facecolor='white')
        axis = labels.add_axes(position)
        axis.set(xlim=camera[:2], ylim=camera[2:], aspect='equal', xlabel='$x/c$', ylabel='$y/c$')
        axis.add_patch(Polygon(polygon/chord, fc='#d1dbe2', ec='#233747', lw=1.2))
        color_axis = labels.add_axes((.905, .19, .013, .57))
        norm = mpl.colors.Normalize(-vorticity_limit, vorticity_limit)
        labels.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdBu_r'), cax=color_axis,
                        extend='both', label=r'$\omega c/U_D$')
        labels.text(.07, .925, 'Airfoil wake: signed vorticity and velocity', fontsize=20)
        labels.text(.07, .84, rf'$\alpha={cfg["angle"]:g}^\circ$, $b/c={cfg["gap_ratio"]:g}$, '
                    rf'$Re_D={nominal_speed*chord/cfg["nu"]:.0f}$, '+cfg['scheme'].replace('_', '-'))
        labels.text(.07, .055, '2D reduced-model exploration; P1 recovery for display; raw P2 gradients for diagnostics.', fontsize=10)
        buffer = io.BytesIO()
        labels.savefig(buffer, format='png', backend='pgf', dpi=dpi)
    background = Image.open(buffer).convert('RGBA')
    fig = Figure(figsize=size, dpi=dpi, facecolor=(0, 0, 0, 0))
    canvas = FigureCanvasAgg(fig)
    axis = fig.add_axes(position)
    axis.set(xlim=camera[:2], ylim=camera[2:], aspect='equal')
    axis.axis('off')
    triangle = mtri.Triangulation(mesh.p[0]/chord, mesh.p[1]/chord, mesh.t.T)
    image = axis.tripcolor(triangle, np.zeros(mesh.p.shape[1]), shading='gouraud',
                          cmap='RdBu_r', vmin=-vorticity_limit, vmax=vorticity_limit)
    arrows = axis.quiver(arrow_points[0]/chord, arrow_points[1]/chord,
                        np.zeros(arrow_points.shape[1]), np.zeros(arrow_points.shape[1]),
                        color='#25343f', width=.0014, scale_units='xy', scale=nominal_speed/.08)
    width, height = canvas.get_width_height()
    if background.size != (width, height):
        background = background.resize((width, height), Image.Resampling.LANCZOS)
    font = ImageFont.truetype(font_manager.findfont('Times New Roman'), int(.16*dpi))
    encoder = None
    if movie:
        encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-vf',
            'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '20', str(output/'wake.mp4')], stdin=subprocess.PIPE)
    rows: list[dict[str, float]] = []
    true_max = 0.
    try:
        with h5py.File(directory/'results.h5', 'r') as data:
            for index, time in enumerate(times):
                full = np.zeros(vb.N)
                u, v = data['snapshots/u'][index], data['snapshots/v'][index]
                for values, ids in zip((u, v), vb.split_indices()):
                    full[ids] = values
                field = vb.interpolate(full)
                gradient = np.asarray(field.grad)
                j = int(np.argmin(abs(nodes-time)))
                metrics = flow_indicators(np.asarray(field)[:, roi], gradient[:, :, roi], vb.dx[roi], mean[j], chord)
                metrics.update(t=float(time), mean_velocity=float(mean[j]), Re_c=float(abs(mean[j])*chord/cfg['nu']))
                fg = foil_basis.interpolate(full).grad
                shear = np.sum(tangent*np.einsum('ijef,jef->ief', fg, normal), axis=0)
                for name, side in [('upper', body_y > 0.), ('lower', body_y < 0.)]:
                    selected = surface_region & side
                    metrics[name+'_minimum_tangential_gradient'] = float(np.min(shear[selected]))
                    reverse = selected & (shear < -1e-4*max(abs(mean[j]), 1e-9)/chord)
                    metrics[name+'_reversed_surface_length'] = float(np.sum(foil_basis.dx[reverse]))
                rows.append(metrics)
                omega = gradient[1, 0]-gradient[0, 1]
                true_max = max(true_max, float(np.max(abs(omega[roi])))*chord/nominal_speed)
                local = np.einsum('eq,keq->ke', omega*pb.dx, phi)
                load_vector = np.bincount(pb.element_dofs.ravel(), weights=local.ravel(), minlength=pb.N)
                recovered = projection.solve(load_vector)*chord/nominal_speed
                image.set_array(recovered)
                arrows.set_UVC(probe @ u, probe @ v)
                canvas.draw()
                frame = background.copy()
                frame.alpha_composite(Image.fromarray(np.asarray(canvas.buffer_rgba())))
                ImageDraw.Draw(frame).text((int(.76*width), int(.07*height)),
                    f't={time:.5g}, Re={metrics["Re_c"]:.0f}', fill='#233747', font=font)
                frame = frame.convert('RGB')
                if index == 0:frame.save(output/'initial.png')
                if index == len(times)-1:frame.save(output/'final.png')
                if encoder is not None:
                    assert encoder.stdin is not None
                    encoder.stdin.write(np.asarray(frame).tobytes())
                if index % 50 == 0:print(f'wake frame {index+1}/{len(times)}', flush=True)
    finally:
        if encoder is not None:
            assert encoder.stdin is not None
            encoder.stdin.close()
            if encoder.wait() != 0:raise RuntimeError('Wake video encoding failed')
    with (output/'wake_history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    tail = times >= times[-1]-.2*(times[-1]-times[0])
    duration = float(times[tail][-1]-times[tail][0]) if np.sum(tail) > 1 else 0.
    lift = forces[:, 1]
    lift_mean = float(np.trapezoid(lift[tail], times[tail])/duration) if duration else float(lift[-1])
    lift_rms = float(np.sqrt(np.trapezoid((lift[tail]-lift_mean)**2, times[tail])/duration)) if duration else 0.
    assert before == {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    result: dict[str, Any] = dict(status='complete', run_id=directory.name, producer=manifest['source'],
        config=cfg, actual_times=times.tolist(), max_abs_r=float(np.max(abs(scalar_r))),
        final_Re_c=rows[-1]['Re_c'], maximum_backflow_area=max(r['backflow_area'] for r in rows),
        maximum_rotating_area=max(r['rotating_area'] for r in rows),
        final_metrics=rows[-1], tail_model_lift_time_mean=lift_mean, tail_model_lift_time_RMS=lift_rms,
        true_maximum_wake_vorticity=true_max, display_vorticity_limit=vorticity_limit,
        diagnostic_region=[1.05, 5., -.7, .5], excluded_surface_ends=[.2, .97],
        inputs_unchanged=True, input_fingerprints=before, source_sha256=digest(Path(__file__)),
        interpretation='Backflow, rotating-gradient areas and shear reversal are separate candidates; '
                       'none alone certifies resolved vortex shedding or 3D Hele-Shaw validity.')
    if movie:
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output/'wake.mp4'), '-f', 'null', '-'], check=True)
    write_json(output/'wake_analysis.json', result)
    return result

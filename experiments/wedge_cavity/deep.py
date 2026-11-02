"""How deep does the lid-driven cell reach? Stratification in tall wedge cavities.

The wedge geometry fixes the depth-to-width ratio: ``H/W = 1/(2 tan(alpha))``, so the
apex angle alone decides whether the cavity is a shallow dish (2a = 120 deg, H/W = 0.29)
or a deep well (2a = 25 deg, H/W = 2.26). This script measures, for a set of completed
runs, how much of that depth the driven cell actually occupies:

``active depth``
    where ``|u|`` on the symmetry axis falls below 1 % of its lid value;
``dead-water fraction``
    share of the area on which ``psi`` stays within 5 % of its minimum;
``reversal count``
    sign changes of ``u(0, y)`` below the lid, i.e. how many distinct cells the axis
    passes through.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import fields, report, study

PROJECT = Path(__file__).resolve().parents[2]
INK = '#1F1F1D'
MUTED = '#8A8A85'
ACCENT = '#185FA5'
WARM = '#C0392B'


def measure(directory: Path) -> dict[str, Any]:
    cfg, manifest, space, model, run = report.open_run(directory)
    index = len(run.snapshot_times) - 1
    state = run.state(index)
    velocity = model.vector(state)
    height, width = space.height, space.half_width

    depth = np.linspace(0., 1., 801)
    points = np.vstack([np.zeros_like(depth), -depth*height])
    u, v = model.evaluate_velocity(velocity, points)
    speed = np.hypot(u, v)
    peak = float(speed.max())

    below = np.where(speed < 0.01*peak)[0]
    active = float(depth[below[0]]) if below.size else 1.0

    # Reversals of the axial velocity: ignore the thin sheet where |u| is numerical noise
    # relative to the lid speed, otherwise round-off near the apex counts as a cell.
    sign = np.sign(u)
    solid = np.abs(u) > 1e-3*peak
    transitions = np.count_nonzero((sign[:-1]*sign[1:] < 0.) & solid[:-1] & solid[1:])

    sample = fields.FieldSampler(model).sample(state)
    psi = sample.streamfunction
    spread = float(np.nanmax(psi)-np.nanmin(psi))
    # Two different questions, so two different numbers: `dead_fraction` is the area the
    # driven cell never reaches (speed criterion, the one the figure shades), while
    # `psi_dead_fraction` is the area where the streamfunction has almost flattened out.
    dead = float(np.count_nonzero(sample.speed < 0.01*float(np.nanmax(sample.speed)))
                 /sample.speed.size)
    psi_dead = float(np.count_nonzero(np.abs(psi-np.nanmin(psi)) < 0.05*spread)/psi.size)

    return {'run': directory.name, 'directory': str(directory),
            'apex_angle': cfg['apex_angle'], 'reynolds': cfg['reynolds'],
            'lid_cells': cfg['lid_cells'], 'height': height, 'half_width': width,
            'depth_ratio': height/width, 'active_depth': active,
            'reversals': int(transitions), 'dead_fraction': dead,
            'psi_dead_fraction': psi_dead,
            'psi_min': float(np.nanmin(psi)), 'psi_spread': spread,
            'depth': depth, 'axial_u': u, 'axial_speed': speed,
            'sample': sample, 'space': space, 'cfg': cfg}


def stratification_figure(records: list[dict], *, output: Path) -> dict:
    """Each cavity drawn in its own (x/W, -y/H) box, so the layers line up.

    Plotting in physical coordinates would make the H/W = 4.5 case eight times taller
    than the H/W = 0.58 one and hide the comparison. Rescaling by the lid half-width
    and the depth puts every cavity on the same unit triangle: the vertical axis is
    then simply the fraction of the depth, which is what the question is about.
    """
    ordered = sorted(records, key=lambda record: record['depth_ratio'])
    figure, axes = plt.subplots(1, len(ordered)+1, figsize=(2.55*(len(ordered)+1), 4.4),
                                layout='constrained', squeeze=False)
    axes = axes[0]
    for position, record in enumerate(ordered):
        ax = axes[position]
        sample, space = record['sample'], record['space']
        X = sample.x/space.half_width
        Y = -sample.y/space.height
        # psi divided by |psi_min|: every cavity then shares one scale, so the panels
        # can be read against each other instead of against six private colour maps.
        normalised = sample.streamfunction/abs(record['psi_min'])
        image = ax.tripcolor(X, Y, sample.triangles, normalised, shading='gouraud',
                             cmap='RdBu_r', vmin=-1., vmax=0.)
        levels = np.linspace(-1., 0., 33)[1:-1]
        ax.tricontour(X, Y, sample.triangles, normalised,
                      levels=levels, colors='#25251F', linewidths=.42, alpha=.45)
        # The dead-water core: everything the driven cell never reaches.
        dead = sample.speed < 0.01*float(np.nanmax(sample.speed))
        ax.tricontourf(X, Y, sample.triangles, dead.astype(float), levels=[.5, 1.5],
                       colors=['#6B6B66'], alpha=.30)
        ax.plot([-1., 0., 1.], [0., 1., 0.], color='#111111', lw=1.2)
        ax.plot([-1., 1.], [0., 0.], color=WARM, lw=2.4, solid_capstyle='butt')
        ax.annotate('', xy=(0.42, -0.055), xytext=(-0.05, -0.055),
                    arrowprops=dict(arrowstyle='->', color=WARM, lw=1.2))
        ax.set_xlim(-1.08, 1.08); ax.set_ylim(1.08, -0.10)
        ax.set_aspect('auto')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(r'$2\alpha$' + f"={record['apex_angle']:.0f}°\n"
                     r'$H/L$' + f"={record['depth_ratio']/2:.2f}", fontsize=9, color=INK, pad=4)
        ax.text(0., 1.055, f"dead water {100*dead.mean():.0f}%", fontsize=7.2,
                color='#4A4A46', ha='center', va='bottom')
        if position == 0:
            ax.set_ylabel(r'$-y/H$', fontsize=9)
            ax.set_yticks([0., .25, .5, .75, 1.])
            ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=7)
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_color(MUTED)

    ax = axes[-1]
    for record in ordered:
        ax.plot(record['axial_u'], record['depth'], lw=1.5,
                label=(r'$2\alpha$' + f"={record['apex_angle']:.0f}°"))
    ax.axvline(0., color=MUTED, ls=':', lw=.9)
    ax.set_ylim(1.05, -0.05)
    ax.set_xlabel(r'$u(0,y)/U_0$', fontsize=9)
    ax.set_ylabel(r'$-y/H$', fontsize=9)
    ax.set_title('axial velocity', fontsize=9.5, color=INK)
    ax.grid(alpha=.22)
    ax.legend(fontsize=7, frameon=False, loc='lower right')

    figure.colorbar(image, ax=axes[:-1], orientation='horizontal', fraction=.035, pad=.02,
                    aspect=70, label=r'$\psi/|\psi_{\min}|$')
    figure.suptitle('lid-driven wedge cavity: the driven cell versus the dead-water core',
                    fontsize=11, color=INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return {'output': str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Stratification of the lid-driven wedge cavity')
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, default=PROJECT/'reports'/'wedge_cavity_deep')
    args = parser.parse_args()
    records = [measure(directory) for directory in args.directories]
    summary: dict[str, Any] = {'table': [], 'figures': {}}
    for record in records:
        summary['table'].append({key: value for key, value in record.items()
                                 if key not in ('depth', 'axial_u', 'axial_speed', 'sample',
                                                'space', 'cfg', 'directory')})
    summary['figures']['stratification'] = stratification_figure(
        records, output=args.out/'stratification.png')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'deep.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                 default=lambda value: None)+'\n')
    print(json.dumps(summary['table'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Plot the lid-driven wedge cavity mesh without assembling or advancing anything."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from solver.fem.wedge import WedgeSpace, triangle_quality

MESH, LID, WALL, APEX = '#185FA5', '#C0392B', '#2B2B2B', '#0F6E56'


def draw(space: WedgeSpace, dpi: int, out: Path) -> dict[str, float]:
    mesh, quality = space.mesh, triangle_quality(space.mesh)
    half_width, height = space.half_width, space.height
    boundaries = mesh.boundaries
    assert boundaries is not None
    x, y = mesh.p
    ends = mesh.p[:, mesh.facets]

    fig = plt.figure(figsize=(13.2, 6.4), layout='constrained')
    grid = fig.add_gridspec(3, 2, width_ratios=[1.42, 1.], height_ratios=[1., 1., .12])
    ax = fig.add_subplot(grid[0:2, 0])
    ax_lid = fig.add_subplot(grid[0, 1])
    ax_apex = fig.add_subplot(grid[1, 1])
    ax_caption = fig.add_subplot(grid[2, :])
    ax_caption.axis('off')

    # --- full mesh -------------------------------------------------------
    ax.triplot(x, y, lw=.32, color=MESH, alpha=.5)
    for group, colour, width in (('lid', LID, 2.6), ('wall_left', WALL, 2.2), ('wall_right', WALL, 2.2)):
        for facet in boundaries[group]:
            ax.plot(ends[0, :, facet], ends[1, :, facet], color=colour, lw=width, solid_capstyle='round')
    ax.plot([], [], lw=2.6, color=LID, label='lid: sliding wall, $u=(U,0)$')
    ax.plot([], [], lw=2.2, color=WALL, label=r'inclined walls: no-slip, $u=0$')

    ax.annotate('', xy=(half_width*.90, .07*height), xytext=(-half_width*.90, .07*height),
                arrowprops=dict(arrowstyle='-|>', color=LID, lw=2.8, mutation_scale=24))
    ax.text(0., .13*height, '$U$', color=LID, ha='center', va='bottom', fontsize=14)

    ax.plot([0.], [-height], marker='o', ms=6.5, mfc=APEX, mec='white', mew=1.3, zorder=6)
    ax.annotate(f'apex, $2\\alpha={space.apex_angle:g}^\\circ$', xy=(0., -height),
                xytext=(-half_width*1.34, -height*.98), fontsize=9.5, color=APEX, va='center',
                arrowprops=dict(arrowstyle='-', color=APEX, lw=1.1, shrinkA=2, shrinkB=4),
                bbox=dict(fc='white', ec='none', alpha=.9, pad=1.5))

    xd = half_width*1.14
    ax.annotate('', xy=(xd, -.004*height), xytext=(xd, -height),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.1, shrinkA=0, shrinkB=0))
    ax.text(xd + half_width*.06, -height/2, '$H$', fontsize=12, color='#333', va='center')

    yd = -height*1.16
    ax.annotate('', xy=(half_width, yd), xytext=(-half_width, yd),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.1, shrinkA=0, shrinkB=0))
    ax.text(0., yd - height*.05, '$L$', fontsize=12, color='#333', ha='center', va='top')

    ax.set_aspect('equal')
    ax.set_xlim(-half_width*1.52, half_width*1.42)
    ax.set_ylim(-height*1.34, height*.26)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Wedge cavity, Taylor–Hood P2/P1: {mesh.t.shape[1]} affine triangles', fontsize=10.5)
    ax.legend(fontsize=8.5, loc='lower left', bbox_to_anchor=(-.02, -.02), framealpha=.94)

    # --- zoom insets -----------------------------------------------------
    span = .16*min(half_width, height)
    for target, centre in ((ax_lid, (half_width, 0.)), (ax_apex, (0., -height))):
        target.triplot(x, y, lw=.8, color=MESH, alpha=.95)
        for group, colour, width in (('lid', LID, 2.0), ('wall_left', WALL, 1.8), ('wall_right', WALL, 1.8)):
            for facet in boundaries[group]:
                target.plot(ends[0, :, facet], ends[1, :, facet], color=colour, lw=width)
        target.set_aspect('equal')
        if target is ax_lid:
            target.set_xlim(centre[0] - span, centre[0] + .04*span)
            target.set_ylim(-span, .18*span)
            target.set_title('zoom: lid corner (singular)', fontsize=9)
        else:
            target.set_xlim(-span, span)
            target.set_ylim(centre[1] - .28*span, centre[1] + span)
            target.set_title('zoom: apex (singular)', fontsize=9)

    # --- caption ---------------------------------------------------------
    caption = (f"$L={space.lid_length:g}$,  $H={height:.4f}$,  "
               f"$h_x\\times h_y={space.element_size[0]:.4f}\\times{space.element_size[1]:.4f}$,  "
               f"{mesh.p.shape[1]} nodes, {mesh.t.shape[1]} triangles,  "
               f"min angle ${quality['min_angle_deg']:.2f}^\\circ$,  "
               f"edge ratio $\\leq{quality['max_aspect']:.3f}$,  "
               f"every triangle congruent (equal area)")
    ax_caption.text(.5, .55, caption, ha='center', va='center', fontsize=9, color='#333')
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apex-angle', type=float, default=90.)
    parser.add_argument('--lid-cells', type=int, default=64)
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--out', type=Path, default=Path('wedge_cavity_mesh.png'))
    args = parser.parse_args()
    space = WedgeSpace(apex_angle=args.apex_angle, lid_cells=args.lid_cells)
    quality = draw(space, args.dpi, args.out)
    print(f"{args.out}  min_angle={quality['min_angle_deg']:.2f}")


if __name__ == '__main__':
    main()

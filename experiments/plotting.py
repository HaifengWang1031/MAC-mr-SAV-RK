"""Legacy common-plot imports plus research-specific reference-error figures.

Common reading/rendering/style has one implementation in postprocess. The fixed
two-scheme, slope-two convergence presentation below remains a research policy.
"""
import numpy as np
from postprocess.mac import read_run_fields as read_run_fields
from postprocess.plotting import (
    FIELD_CMAPS as FIELD_CMAPS,
    LATEX_SEARCH_PATHS as LATEX_SEARCH_PATHS,
    field_figure as field_figure,
    history_figure as history_figure,
    latex_binary as latex_binary,
    latex_style as latex_style,
    usetex_available as usetex_available,
)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def reference_error_figure(rows: list[dict[str, str]]) -> Figure:
    """Saved L2/H1 errors, with gaps and axis markers instead of fake failed errors."""
    times = sorted({float(row['time']) for row in rows})
    figure, axes = plt.subplots(2, len(times), figsize=(4.4*len(times), 7), squeeze=False,
                                constrained_layout=True)
    for column, time in enumerate(times):
        for line, norm in enumerate(('L2', 'H1')):
            axis = axes[line, column]
            guide = None
            for scheme, color, marker in [('sdirk2', '#0072B2', 'o'),
                                           ('sdirk2_mrsav', '#D55E00', '^')]:
                data = sorted([r for r in rows if r['scheme'] == scheme and float(r['time']) == time],
                              key=lambda r: float(r['dt']))
                dt = np.array([float(r['dt']) for r in data])
                errors = np.array([float(r[norm]) if r['status'] == 'complete' else np.nan for r in data])
                errors[errors <= 0] = np.nan
                axis.loglog(dt, errors, marker=marker, color=color, markerfacecolor='none', label=scheme)
                failed = np.array([r['status'] != 'complete' for r in data], dtype=bool)
                if failed.any():
                    axis.plot(dt[failed], np.full(failed.sum(), .025), 'x', color=color,
                              transform=axis.get_xaxis_transform(), label=scheme+' failed (no error)')
                valid = np.isfinite(errors)
                if guide is None and valid.any():
                    guide = dt[valid][0], errors[valid][0]
            if guide is not None:
                domain = np.array(sorted({float(r['dt']) for r in rows}))
                axis.loglog(domain, guide[1]*(domain/guide[0])**2, '--', color='0.45', label='slope 2 guide')
            axis.set(xlabel='time step', ylabel='L2 error' if norm == 'L2' else 'H1 seminorm error',
                     title=f't = {time:g}')
            axis.grid(alpha=.25, which='both')
            axis.legend(fontsize=7)
    return figure

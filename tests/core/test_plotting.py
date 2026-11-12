"""Shared readonly rendering and external-tool style without research imports."""
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('toolchain', [False, True])
def test_latex_style_locates_tools_and_restores_state_even_on_failure(tmp_path: Path, monkeypatch,
                                                                      toolchain: bool) -> None:
    from postprocess import plotting
    import matplotlib.pyplot as plt

    bindir = tmp_path/'texbin'
    bindir.mkdir()
    if toolchain:
        for name in ('latex', 'dvipng'):
            executable = bindir/name
            executable.write_text('#!/bin/sh\nexit 1\n')
            executable.chmod(0o755)
    monkeypatch.setenv('PATH', '')
    monkeypatch.setattr(plotting, 'LATEX_SEARCH_PATHS', (str(bindir),))
    assert plotting.latex_binary('latex') == (str(bindir/'latex') if toolchain else None)
    assert plotting.usetex_available() == toolchain
    # A caller's previous TeX setting must not defeat the documented fallback.
    with plt.rc_context({'text.usetex': True}):
        old_size = plt.rcParams['axes.labelsize']
        with pytest.raises(RuntimeError, match='test interruption'):
            with plotting.latex_style({'axes.labelsize': 17}):
                assert plt.rcParams['axes.labelsize'] == 17
                assert plt.rcParams['text.usetex'] == toolchain
                assert os.environ['PATH'] == (str(bindir)+os.pathsep if toolchain else '')
                raise RuntimeError('test interruption')
        assert plt.rcParams['axes.labelsize'] == old_size
        assert plt.rcParams['text.usetex'] is True
        assert os.environ['PATH'] == ''


def test_history_figure_handles_one_recorded_panel_and_closes_its_figure(tmp_path: Path) -> None:
    from postprocess.plotting import history_figure
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    destination = tmp_path/'history.png'
    history_figure({'time': np.array([0., .25]), 'r': np.array([0., .01])}, destination)
    assert destination.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert plt.get_fignums() == before


@pytest.mark.parametrize('kind', ['field', 'history'])
def test_public_figures_fall_back_without_tex_and_restore_callers_choice(tmp_path: Path, monkeypatch,
                                                                        kind: str) -> None:
    from postprocess import plotting
    from solver.mac.grid import MACGrid
    import matplotlib.pyplot as plt

    monkeypatch.setenv('PATH', '')
    monkeypatch.setattr(plotting, 'LATEX_SEARCH_PATHS', ())
    destination = tmp_path/f'{kind}.png'
    before = plt.get_fignums()
    try:
        with plt.rc_context({'text.usetex': True}):
            if kind == 'field':
                plotting.field_figure(MACGrid(4, 3), [0.], {0.: np.ones((3, 4))}, destination, 'speed')
            else:
                plotting.history_figure({'time': np.array([0., .25]), 'r': np.array([0., .01])}, destination)
            assert plt.rcParams['text.usetex'] is True
            assert os.environ['PATH'] == ''
        assert destination.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert plt.get_fignums() == before
    finally:
        for number in set(plt.get_fignums())-set(before):
            plt.close(number)

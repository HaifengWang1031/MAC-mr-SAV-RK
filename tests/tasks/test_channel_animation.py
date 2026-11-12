"""Saved-field observables, periodic tracer advection and read-only movie export."""
import json
import shutil

import numpy as np
import pytest

from experiments.channel_ns.animate import read_movie, render, sample_velocity, tracer_paths
from experiments.channel_ns.run import run
from experiments.workflow import digest
from solver.channel import ChannelGrid


def test_equilibrium_movie_observables_and_known_particle_advection(tmp_path):
    directory = run({'case': 'poiseuille', 'backend': 'fft', 'scheme': 'sdirk3',
                     'nx': 12, 'ny': 8, 'lx': 2., 'ly': 1., 'nu': .1, 'driving': .2,
                     'T': .1, 'dt': .01, 'snapshots': [0., .05, .1]}, root=tmp_path)
    data = read_movie(directory)
    grid = ChannelGrid(12, 8, 2., 1.)
    # Independently known discrete half-cell Poiseuille correction F*hy²/(8*nu).
    yc = (np.arange(grid.ny)+.5)*grid.hy
    reference = .2*yc*(1-yc)/(.2)+.2*grid.hy**2/(.8)
    np.testing.assert_allclose(data.profiles, np.tile(reference, (3, 1)), atol=2e-14)
    assert data.perturbation_norm.max() < 1e-13
    assert np.max(np.abs(data.perturbation_vorticity)) < 1e-12
    paths = tracer_paths(data, count=12)
    velocity = sample_velocity(grid, data.u[0], data.v[0], paths[0])
    np.testing.assert_allclose(paths[:, :, 1], np.tile(paths[0, :, 1], (3, 1)), atol=1e-14)
    np.testing.assert_allclose(paths[-1, :, 0], (paths[0, :, 0]+.1*velocity[:, 0]) % grid.lx, atol=1e-13)
    wall = sample_velocity(grid, data.u[0], data.v[0], np.array([[0., 0.], [2., 1.]]))
    np.testing.assert_array_equal(wall, 0.)


@pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('xelatex') is None,
                    reason='optional movie export tools unavailable')
def test_movie_export_uses_saved_fields_without_running_solver(tmp_path, monkeypatch):
    directory = run({'case': 'perturbed', 'backend': 'fft', 'scheme': 'sdirk3',
                     'nx': 12, 'ny': 8, 'lx': 3., 'ly': 1.,
                     'T': .02, 'dt': .01, 'snapshots': [0., .01, .02]}, root=tmp_path)
    import experiments.channel_ns.run as runner
    monkeypatch.setattr(runner, 'build_model', lambda cfg: (_ for _ in ()).throw(
        AssertionError('Animation must not launch computation')))
    report = render(directory, root=tmp_path, fps=10, particles=12, dpi=60)
    summary = json.loads((report/'animation.json').read_text())
    assert summary['status'] == 'complete' and summary['frames'] == 3
    assert summary['actual_times'] == [0., .01, .02]
    for name in ('channel_flow.mp4', 'channel_flow.gif', 'frame_0000.png', 'frame_0002.png'):
        assert (report/name).stat().st_size > 1000


@pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('xelatex') is None,
                    reason='optional movie export tools unavailable')
def test_near_wall_ts_export_uses_separate_output_and_preserves_default_report(tmp_path):
    directory = run({'case': 'ts_wave', 'backend': 'fft', 'scheme': 'sdirk3',
                     'nx': 12, 'ny': 20, 'lx': 2*np.pi/.8, 'ly': 2.,
                     'nu': 1/30000, 'driving': 2/30000, 'epsilon': .025,
                     'T': .02, 'dt': .01, 'snapshots': [0., .01, .02]}, root=tmp_path)
    original = render(directory, root=tmp_path, fps=10, particles=12, dpi=60)
    original_hash = digest(original/'animation.json')
    target = tmp_path/'near_wall_export'
    exported = render(directory, root=tmp_path, fps=10, particles=12, dpi=60,
                      near_wall=True, output_directory=target)
    saved = json.loads((exported/'animation.json').read_text())
    assert exported == target and digest(original/'animation.json') == original_hash
    assert saved['status'] == 'complete' and saved['options']['near_wall']
    assert saved['vorticity_y_limits'] == [0., .4] and saved['vertical_zoom'] == 5.
    assert saved['reynolds_half_height'] == 30000 and saved['initial_peak_perturbation_fraction'] == .025

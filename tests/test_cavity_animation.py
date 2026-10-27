"""Moving-wall tracer interpolation, saved MAC observables and read-only export."""
import json
import shutil

import numpy as np
import pytest

from experiments.cavity.animate import read_movie, render, sample_velocity
from experiments.workflow import run_experiment
from solver.mac.grid import MACGrid


def test_face_interpolation_respects_independent_couette_profile_and_walls():
    grid = MACGrid(12, 10)
    _, y = grid.coordinates('u')
    u = y.copy()
    u[:, [0, -1]] = 0.
    v = np.zeros((grid.ny+1, grid.nx))
    points = np.array([[.4, .03], [.55, .31], [.62, .93]])
    np.testing.assert_allclose(sample_velocity(grid, u, v, points, 1.),
                               np.column_stack([points[:, 1], np.zeros(3)]), atol=1e-15)
    walls = np.array([[.4, 0.], [0., .6], [1., .6], [.4, 1.]])
    np.testing.assert_allclose(sample_velocity(grid, u, v, walls, 1.),
                               [[0., 0.], [0., 0.], [0., 0.], [1., 0.]], atol=1e-15)


@pytest.mark.skipif(any(shutil.which(tool) is None for tool in ('ffmpeg', 'xelatex', 'pdffonts')),
                    reason='optional animation export tools unavailable')
def test_cavity_movie_reads_saved_fields_without_running_the_solver(tmp_path, monkeypatch):
    directory = run_experiment({'experiment': 'cavity', 'nx': 8, 'ny': 8, 'nu': .01,
        'lid_speed': 1., 'T': .02, 'dt': .01, 'scheme': 'sdirk2_mrsav',
        'snapshots': [0., .01, .02]}, root=tmp_path)
    data = read_movie(directory)
    np.testing.assert_array_equal(data.speed[0], 0.)
    np.testing.assert_allclose(data.streamfunction[:, -1], 0., atol=1e-12)
    assert data.max_divergence < 1e-12
    monkeypatch.setattr('experiments.workflow.run_experiment', lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError('Animation launched a computation')))
    report = render(directory, root=tmp_path, fps=10, particles=12, dpi=60)
    record = json.loads((report/'animation.json').read_text())
    assert record['status'] == 'complete' and record['frames'] == 3 and record['font_verified']
    assert record['actual_times'] == [0., .01, .02] and record['tracer_boundary_clips'] == 0
    for name in ('cavity_flow.mp4', 'cavity_flow.gif', 'frame_0000.png', 'frame_0002.png'):
        assert (report/name).stat().st_size > 1000
    (directory/'results.h5').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        read_movie(directory)

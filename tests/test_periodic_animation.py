"""Independent shear reference, periodic tracers, and saved-field-only movie export."""
from dataclasses import replace
import json
import shutil

import numpy as np
import pytest

from experiments.periodic_ns.animate import read_movie, render, sample_velocity, tracer_paths
from experiments.periodic_ns.run import run


def test_periodic_observables_match_known_driven_shear_and_tracers_wrap(tmp_path):
    directory = run({'case': 'kolmogorov', 'scheme': 'sdirk3', 'nx': 18, 'ny': 18,
        'nu': .1, 'amplitude': .2, 'k_f': 2, 'epsilon': 0., 'T': .06, 'dt': .01,
        'snapshots': [0., .03, .06]}, root=tmp_path)
    data = read_movie(directory)
    y = 2*np.pi*np.arange(18)/18
    amplitude = .2/(.1*2**2)*(1-np.exp(-.1*2**2*data.times))
    expected = -amplitude[:, None, None]*np.sin(2*y)[None, :, None]*np.ones((1, 1, 18))
    np.testing.assert_allclose(data.u, expected, atol=3e-10, rtol=0.)
    np.testing.assert_allclose(data.v, 0., atol=1e-14)
    np.testing.assert_allclose(data.vorticity,
        2*amplitude[:, None, None]*np.cos(2*y)[None, :, None]*np.ones((1, 1, 18)), atol=6e-10, rtol=0.)
    np.testing.assert_allclose(4*data.streamfunction, data.vorticity, atol=1e-13)
    translation = np.array([120., -120.])  # Cross each periodic boundary within this short interval.
    translated = replace(data, u=np.full_like(data.u, translation[0]), v=np.full_like(data.v, translation[1]))
    paths = tracer_paths(translated, count=24)
    for index, time in enumerate(data.times):
        np.testing.assert_allclose(paths[index], (paths[0]+time*translation) % (2*np.pi), atol=2e-14)
    points = np.array([[-.1, 7.], [2*np.pi, 0.]])
    np.testing.assert_allclose(sample_velocity(translated.u[0], translated.v[0], points, 2*np.pi, 2*np.pi),
                              np.tile(translation, (2, 1)), atol=1e-13)


@pytest.mark.skipif(any(shutil.which(tool) is None for tool in ('ffmpeg', 'xelatex', 'pdffonts')),
                    reason='optional animation export tools unavailable')
def test_periodic_movie_export_uses_saved_fields_without_running_solver(tmp_path, monkeypatch):
    directory = run({'case': 'kolmogorov', 'scheme': 'sdirk3', 'nx': 18, 'ny': 18,
        'nu': .1, 'amplitude': .2, 'k_f': 2, 'epsilon': .1, 'T': .02, 'dt': .01,
        'snapshots': [0., .01, .02]}, root=tmp_path)
    monkeypatch.setattr('experiments.periodic_ns.run.build_model', lambda *a, **kw:
                        (_ for _ in ()).throw(AssertionError('Animation launched a solver')))
    report = render(directory, root=tmp_path, fps=10, particles=12, dpi=60)
    result = json.loads((report/'animation.json').read_text())
    assert result['status'] == 'complete' and result['font_verified'] and result['frames'] == 3
    assert result['actual_times'] == [0., .01, .02]
    for name in ('kolmogorov_flow.mp4', 'kolmogorov_flow.gif', 'frame_0000.png', 'frame_0002.png'):
        assert (report/name).stat().st_size > 1000
    (directory/'results.h5').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        read_movie(directory)

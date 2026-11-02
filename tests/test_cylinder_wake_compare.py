"""SAV-only comparison, saved mesh/field checks, sensitivity norms and readonly movies."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.cylinder_ns.run import run
from experiments.cylinder_wake_compare.compute import prepare
from experiments.cylinder_wake_compare.checks import _sample, verify
from experiments.cylinder_wake_compare.analyze import analyze
from experiments.cylinder_wake_compare.preview import preview
from experiments.workflow import digest, write_json


@pytest.mark.skipif(any(shutil.which(name) is None for name in ('ffmpeg', 'xelatex', 'pdffonts')),
                    reason='optional movie tools unavailable')
def test_saved_sav_comparison_checks_and_readonly_movie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = prepare({'resolution': .25, 'circle_segments': 16, 'T': .02001, 'display_start': .00001,
                         'snapshot_interval': .01, 'check_duration': .005, 'fps': 10}, root=tmp_path)
    manifest = json.loads((directory/'manifest.json').read_text())
    for case in manifest['cases']:
        folder = Path(case['directory'])
        output = run(json.loads((folder/'input.json').read_text()), root=folder)
        saved = json.loads((output/'manifest.json').read_text())
        assert saved['status'] == 'complete'
        write_json(folder/'result.json', {'status': 'complete', 'run': str(output),
                                         'results_sha256': saved['results_sha256']})
        case['status'] = 'complete'
    assert manifest['preparation_source']['code_sha256'] == saved['source']['code_sha256']
    manifest['status'] = 'computed'
    write_json(directory/'manifest.json', manifest)
    verify(directory)
    manifest.update(status='complete', checks_sha256=digest(directory/'checks.json'))
    write_json(directory/'manifest.json', manifest)
    from solver.fem import FEMModel, OpenFlowModel
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError('Readonly analysis constructed a flow model')
    monkeypatch.setattr(FEMModel, '__init__', forbidden)
    monkeypatch.setattr(OpenFlowModel, '__init__', forbidden)
    report = analyze(directory, root=tmp_path, dpi=60)
    summary = json.loads((report/'analysis.json').read_text())
    assert summary['status'] == 'complete' and summary['frames'] == 3 and summary['font_verified']
    assert summary['flow_axes_use_equal_scale']
    assert len(summary['short_sensitivity']['rows']) == 2
    assert all(item['config']['scheme'] == 'sdirk3_mrsav' for item in summary['cases'])
    for name in ('cylinder_wakes.mp4', 'cylinder_wakes.gif', 'preview.gif', 'frame_0002.png'):
        assert (report/name).stat().st_size > 1000
    raw = Path(json.loads((Path(manifest['cases'][0]['directory'])/'result.json').read_text())['run'])
    saved_preview = preview(raw, root=tmp_path, start=.00001, fps=10, dpi=60)
    preview_summary = json.loads((saved_preview/'analysis.json').read_text())
    assert preview_summary['status'] == 'complete' and preview_summary['frames'] == 3
    assert preview_summary['actual_times'][0] < .00001
    assert preview_summary['font_verified'] and preview_summary['flow_axes_use_equal_scale']
    original_preview = digest(saved_preview/'analysis.json')
    smoothed = preview(raw, root=tmp_path, start=.00001, fps=10, dpi=60, smooth=True)
    smoothed_summary = json.loads((smoothed/'analysis.json').read_text())
    assert smoothed != saved_preview and smoothed_summary['status'] == 'complete'
    assert smoothed_summary['frames'] == 3 and smoothed_summary['visualization']['cylinder_masked']
    assert smoothed_summary['visualization']['temporal_interpolation'] is False
    assert digest(saved_preview/'analysis.json') == original_preview
    (directory/'checks.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        analyze(directory, root=tmp_path, dpi=60)


def test_compare_rejects_non_sav_and_mismatched_display_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='SAV'):
        prepare({'scheme': 'sdirk3'}, root=tmp_path)
    with pytest.raises(ValueError, match='equally spaced'):
        prepare({'T': 12., 'display_start': 10., 'snapshot_interval': .17}, root=tmp_path)


def test_bounded_fem_sampling_preserves_an_independent_quadratic_field() -> None:
    from solver.fem import CylinderSpace
    basis, _ = CylinderSpace(.25, 16, 6).bases()
    def field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([1+x+2*y+x*y+2*x*x-3*y*y, 2-x+3*y+x*y-y*y])
    full = np.zeros(basis.N)
    for component, indices in enumerate(basis.split_indices()):
        full[indices] = field(*basis.doflocs[:, indices])[component]
    points = basis.global_coordinates().reshape(2, -1)[:, ::7]
    assert points.shape[1] > 128
    np.testing.assert_allclose(_sample(basis, full, points), field(*points), atol=2e-13, rtol=0.)


@pytest.mark.parametrize('sigma,quadratic', [(0., True), (.008, False)])
def test_continuous_display_recovers_known_curl_and_masks_body(sigma: float, quadratic: bool) -> None:
    from solver.fem import CylinderSpace
    from experiments.cylinder_wake_compare.smooth import SmoothVorticityGrid
    basis, _ = CylinderSpace(.25, 16, 6).bases()
    ui, vi = basis.split_indices()
    u = -basis.doflocs[1, ui]
    v = basis.doflocs[0, vi].copy()
    if quadratic:
        u = u+2*basis.doflocs[1, ui]**2
        v = v+1.5*basis.doflocs[0, vi]**2
    before_u, before_v = u.copy(), v.copy()
    display = SmoothVorticityGrid(basis, bounds=(0., 2.2, 0., .41), width=220, sigma=sigma)
    values = display.frame(u, v)
    x, y = np.meshgrid(display.x, display.y)
    expected = 2.+3*x-4*y if quadratic else np.full(x.shape, 2.)
    np.testing.assert_allclose(values.compressed(), expected[~display.mask], rtol=0., atol=5e-11)
    assert values.mask[np.argmin(abs(display.y-.2)), np.argmin(abs(display.x-.2))]
    assert np.isfinite(values.compressed()).all()
    np.testing.assert_array_equal(u, before_u)
    np.testing.assert_array_equal(v, before_v)

"""A real small step scan retains controls, outcomes and truthful qualification."""
import json
from pathlib import Path

import pytest
import numpy as np

from experiments.records import digest
from tools.rerun_low_high_stability_probe import baseline_config
from tools.search_low_high_stability_steps import search

# The archived baseline record this probe replays is external, untracked evidence.
_BASELINE = (Path(__file__).resolve().parents[1]/'runs/low_high_stability_probe'
             /'20261004T080538558295'/'metadata.json')
pytestmark = pytest.mark.skipif(
    not _BASELINE.is_file(),
    reason=f'missing archived baseline record {_BASELINE.relative_to(Path(__file__).resolve().parents[1])}')


def test_search_passes_steps_to_saved_runs_and_preserves_unqualified_controls(tmp_path):
    cfg = {**baseline_config(), 'nx': 12, 'T': .02}
    folder = search(cfg, [.004, .008], tmp_path, workers=2)
    record = json.loads((folder/'search.json').read_text())
    assert record['status'] == 'complete' and len(record['cases']) == 2
    assert record['best'] is None and record['failed_case_count'] == 0
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['search_sha256'] == digest(folder/'search.json')
    for row in record['cases']:
        assert not row['qualifies']
        assert row['ordinary_status'] == row['sav_status'] == 'complete'
        run = folder/row['run']
        saved = json.loads((run/'config.json').read_text())
        assert saved == {**cfg, 'dt': row['dt']}
        index = json.loads((run/'archive_manifest.json').read_text())
        for member in index['members']:
            with np.load(run/f'{member}.npz') as data:
                times = data['history'][:, 0]
                assert len(times)-1 == int(np.ceil(cfg['T']/row['dt']))
                assert times[-1] == cfg['T']
                np.testing.assert_allclose(np.diff(times)[:-1], row['dt'], atol=1e-16)

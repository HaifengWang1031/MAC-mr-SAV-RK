"""Focused checks for noninteger levels and the four-scheme report layout."""
import json
import numpy as np
import pytest

from experiments.no_slip_reference_convergence.run import schedule, validate
from experiments.no_slip_reference_convergence.analyze import table


def test_fractional_schedule_hits_each_observation_node():
    target = .1*2.**(-2.4)
    steps = schedule([.5, 1., 2., 4.], target)
    nodes = np.r_[0., np.cumsum(steps)]
    for time in (.5, 1., 2., 4.):
        assert np.min(np.abs(nodes-time)) < 1e-12
    assert max(steps) <= target*(1+1e-12)
    assert np.count_nonzero(np.asarray(steps) < target*(1-1e-12)) <= 4


def test_configuration_and_table_have_four_observation_groups():
    from pathlib import Path
    config = json.loads(Path('experiments/no_slip_reference_convergence/configs/production.json').read_text())
    validate(config)
    rows = []
    for level, error in [(2., 1.), (2.2, .75)]:
        for time in config['base']['snapshots']:
            for scheme in ('sdirk2', 'sdirk2_mrsav', 'sdirk3', 'sdirk3_mrsav'):
                rows.append({'k': level, 'time': time, 'scheme': scheme,
                             'L2': error, 'L2_rate': float('nan')})
    rendered = table(rows, config['base']['snapshots'], ('sdirk2', 'sdirk2_mrsav'),
                     'L2', .1, .1*2.**(-10), 128, 128)
    assert rendered.count(r'\multicolumn{4}{c}{$T=') == 4
    assert '2.2 &' in rendered
    assert 'SDIRK2-mr-ccSAV' in rendered
    with pytest.raises(ValueError, match='finer'):
        validate({**config, 'reference_k': 7})

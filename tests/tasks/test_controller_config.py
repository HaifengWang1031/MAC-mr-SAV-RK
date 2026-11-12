"""Experiment configuration rejects noninteger trial limits consistently."""
import pytest

from experiments.channel_ns.run import effective_config as channel_config
from experiments.cylinder_ns.config import effective_config as cylinder_config
from experiments.fem_ns.run import effective_config as fem_config
from experiments.hele_shaw_airfoil.config import effective_config as hele_config
from experiments.periodic_ns.run import effective_config as periodic_config


@pytest.mark.parametrize('configure', [periodic_config, channel_config, fem_config,
                                       cylinder_config, hele_config])
@pytest.mark.parametrize('key', ['max_steps', 'max_rejections'])
@pytest.mark.parametrize('value', [1.5, True])
def test_adaptive_configuration_requires_integer_trial_limits(configure, key, value):
    with pytest.raises(ValueError, match=key):
        configure({'mode': 'adaptive', 'controller': {key: value}})

"""Independent analytic waveforms verify phase-free developed-cycle measurements."""
import numpy as np
import pytest

from experiments.cylinder_ns.benchmark import cycle_statistics, reference_comparison


def _waves():
    t = np.linspace(0., 3., 5001)
    t[1:-1] += .0001*np.sin(np.arange(len(t)-2))
    phase = 2*np.pi*t/.33125
    return t, 3.2+.03*np.cos(2*phase), -.02+np.sin(phase), 2.46+.04*np.sin(2*phase)


def test_irregular_samples_recover_known_lift_period_and_force_extrema():
    t, drag, lift, pressure = _waves()
    result = cycle_statistics(t, drag, lift, pressure, mean_velocity=1., diameter=.1)
    assert result['developed'] and result['cycles'] == 3
    assert result['period'] == pytest.approx(.33125, abs=3e-8)
    assert result['strouhal'] == pytest.approx(.1/.33125, abs=3e-8)
    for key, value in {'drag_min': 3.17, 'drag_max': 3.23, 'drag_mean': 3.2,
                       'lift_min': -1.02, 'lift_max': .98, 'lift_mean': -.02,
                       'pressure_min': 2.42, 'pressure_max': 2.50}.items():
        assert result[key] == pytest.approx(value, abs=3e-6)
    assert reference_comparison(result, {'nu': .001, 'inlet_max': 1.5, 'r0': 0.}, complete=True)['passed']
    assert not reference_comparison(result, {'nu': .001, 'inlet_max': 1.5, 'r0': 0.}, complete=False)['passed']


@pytest.mark.parametrize('kind', ['amplitude_growth', 'period_drift', 'too_short'])
def test_startup_or_insufficient_cycles_cannot_be_qualified(kind: str):
    t, drag, lift, pressure = _waves()
    if kind == 'amplitude_growth':
        lift = -.02+(1+.08*t)*np.sin(2*np.pi*t/.33125)
    elif kind == 'period_drift':
        lift = -.02+np.sin(2*np.pi*(t+.03*t*t)/.33125)
    else:
        t, drag, lift, pressure = (a[t < .5] for a in (t, drag, lift, pressure))
    result = cycle_statistics(t, drag, lift, pressure, mean_velocity=1., diameter=.1)
    assert not result['developed']
    assert not reference_comparison(result, {'nu': .001, 'inlet_max': 1.5, 'r0': 0.}, complete=True)['passed']

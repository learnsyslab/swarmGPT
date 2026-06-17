"""Unit tests for field (D) generators (WS1)."""

import numpy as np

from swarm_gpt.core.fields import CANONICAL_SINE, sine_field


def test_canonical_sine_quarter_error_is_pinned():
    # The frozen quarter approximates sin(pi*u/2) on [0,1] to well under field tolerance.
    from swarm_gpt.core.spline import Spline

    q = Spline(CANONICAL_SINE[:, None], t0=0.0, t1=1.0)
    u = np.linspace(0.0, 1.0, 200)
    err = float(np.max(np.abs(q.evaluate(u)[:, 0] - np.sin(np.pi * u / 2))))
    assert err < 0.015  # < 1.5 % of unit amplitude -> sub-mm at field amplitudes


def test_sine_field_oscillates_with_amplitude_and_phase():
    homes = np.zeros((2, 3))
    amp = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])  # 10 cm vertical
    phase = np.array([0.0, np.pi / 2])
    field = sine_field(homes, t0=0.0, t1=4.0, periods=1, amplitude=amp, phase=phase)
    assert set(field) == {0, 1}
    # Drone 0: sin phase -> starts near 0; drone 1: +pi/2 -> starts near peak.
    np.testing.assert_allclose(field[0].evaluate(0.0), [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(field[1].evaluate(0.0)[2], 10.0, atol=0.2)
    # Peak magnitude stays within amplitude.
    zs = field[0].evaluate(np.linspace(0.0, 4.0, 80))[:, 2]
    assert zs.max() <= 10.0 + 1e-6 and zs.min() >= -10.0 - 1e-6


def test_sine_field_is_purely_in_the_amplitude_direction():
    homes = np.zeros((1, 3))
    amp = np.array([[0.0, 0.0, 5.0]])
    field = sine_field(homes, 0.0, 4.0, periods=2, amplitude=amp, phase=np.zeros(1))
    xy = field[0].evaluate(np.linspace(0.0, 4.0, 50))[:, :2]
    np.testing.assert_allclose(xy, 0.0, atol=1e-9)

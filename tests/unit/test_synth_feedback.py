import re

import pytest

from swarm_gpt.synth.feedback import render

METRICS = {
    "n_drones": 10,
    "duration_s": 12.0,
    "n_steps": 120,
    "min_sep_m": 0.18,
    "min_sep_norm": 0.72,
    "required_sep_m": 0.25,
    "worst_pair": (3, 7),
    "worst_time_s": 4.2,
    "next_worst_norm": 2.1,
    "steps_inside_envelope": 14,
    "authored_min_sep_norm": 0.41,
    "deviation_mean_m": 0.42,
    "deviation_max_m": 0.91,
    "deviation_per_drone_max_m": [0.1] * 10,
    "deviation_worst_drone": 8,
    "failed_solves": 3,
    "max_speed_mps": 0.91,
    "max_accel_mps2": 2.7,
    "min_z_m": 0.5,
    "vel_max_mps": 1.73,
}

# A magnitude with a unit, or a bare decimal: what the relative arm must never emit.
_MAGNITUDE = re.compile(r"\d+\.\d|\d+\s*(m\b|m/s|s\b)")


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="Unknown feedback arm"):
        render("magnitudes", METRICS)


def test_categorical_names_who_and_when_but_no_magnitude():
    text = render("categorical", METRICS)
    assert "Drones 3 and 7" in text
    assert "0.18" not in text and "0.42" not in text


def test_absolute_reports_the_measured_magnitudes():
    text = render("absolute", METRICS)
    assert "0.18 m" in text
    assert "0.25 m" in text
    assert "0.42 m" in text and "0.91 m" in text
    assert "14 of 120" in text


def test_relative_carries_the_ordering_without_any_magnitude():
    text = render("relative", METRICS)
    body = text
    assert not _MAGNITUDE.search(body), f"relative arm leaked a magnitude: {body!r}"
    # 0.41 of the required separation, and 2.1 / 0.72 ~ 2.9x worse than the next worst moment.
    assert "about half of" in body
    assert "roughly three times" in body
    assert "drones 3 and 7" in body


def test_relative_degrades_gracefully_when_nothing_needed_repair():
    clean = METRICS | {
        "steps_inside_envelope": 0,
        "failed_solves": 0,
        "deviation_mean_m": 0.0,
        "deviation_max_m": 0.0,
    }
    text = render("relative", clean)
    assert "none" in text
    assert not _MAGNITUDE.search(text)

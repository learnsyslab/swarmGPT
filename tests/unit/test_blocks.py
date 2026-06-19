"""Unit tests for the block composition evaluator and primitives (WS1)."""

import numpy as np
import pytest

from swarm_gpt.core.blocks import (
    SPLINE_PRIMITIVES,
    breathe,
    cascade,
    center,
    compose_block,
    constant_spline,
    form_circle,
    form_cone,
    form_star,
    grid_form,
    helix,
    helix_form,
    line_form,
    move,
    move_z,
    orbit,
    polygon_form,
    pulse,
    ripple,
    rotate,
    scale,
    shear,
    spiral,
    spline_primitive_by_name,
    swap,
    translate,
    traveling_wave,
    tumble,
    twist,
    twister,
    vee_form,
    wave,
    zig_zag,
)
from swarm_gpt.core.spline import PiecewiseSpline, Spline

_LIMITS = {"lower": np.array([-2.0, -2.0, 0.0]), "upper": np.array([2.0, 2.0, 2.0])}


def _swarm(n: int) -> np.ndarray:
    return np.array([[i * 30.0, 0.0, 100.0] for i in range(n)])


# --- Task 5: block composition evaluator ---------------------------------------------


def test_constant_spline_holds():
    s = constant_spline(np.array([1.0, 2.0, 3.0]), 0.0, 5.0)
    np.testing.assert_allclose(s.evaluate(5.0), [1.0, 2.0, 3.0])


def test_compose_identity_holds_at_home():
    homes = {0: np.array([0.0, 0.0, 100.0]), 1: np.array([50.0, 0.0, 100.0])}
    out = compose_block(homes, [], None, 0.0, 4.0)
    np.testing.assert_allclose(out[1].evaluate(2.0), [50.0, 0.0, 100.0])


def test_compose_adds_piecewise_field_onto_home():
    homes = {0: np.array([0.0, 0.0, 100.0])}
    seg = Spline(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]), 0.0, 2.0)
    seg2 = Spline(np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]]), 2.0, 4.0)
    field = {0: PiecewiseSpline([seg, seg2])}
    out = compose_block(homes, [field], None, 0.0, 4.0)
    np.testing.assert_allclose(out[0].evaluate(2.0), [0.0, 0.0, 110.0])
    np.testing.assert_allclose(out[0].evaluate(0.0), [0.0, 0.0, 100.0])


def test_compose_stacks_two_fields_additively():
    from swarm_gpt.core.fields import sine_field

    home = np.array([10.0, 20.0, 100.0])
    homes = {0: home}
    bump = Spline(np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]), 0.0, 4.0)
    layer_a = {0: bump}
    layer_b = {0: bump}
    out = compose_block(homes, [layer_a, layer_b], None, 0.0, 4.0)
    np.testing.assert_allclose(out[0].evaluate(2.0), home + np.array([0.0, 0.0, 20.0]))

    swarm = np.array([[10.0, 20.0, 100.0]])
    amp1 = np.array([[0.0, 0.0, 25.0]])
    amp2 = np.array([[0.0, 0.0, 15.0]])
    f1 = sine_field(swarm, 0.0, 4.0, 2, amp1, np.zeros(1))
    f2 = sine_field(swarm, 0.0, 4.0, 2, amp2, np.zeros(1))
    stacked = compose_block({0: swarm[0]}, [f1, f2], None, 0.0, 4.0)
    t = 0.7
    expected = swarm[0] + f1[0].evaluate(t) + f2[0].evaluate(t)
    np.testing.assert_allclose(stacked[0].evaluate(t), expected, atol=1e-9)


# --- Task 6: formation primitives ----------------------------------------------------


def _all_hold(out: dict) -> None:
    for s in out.values():
        np.testing.assert_allclose(s.evaluate(s.t0), s.evaluate(s.t1))


def test_formations_return_holds():
    _all_hold(form_circle(([...], 100, 150, 2.0), _swarm(6), 0.0, 6.0, _LIMITS, None))
    _all_hold(form_star((150, 60, 40, 3.0), _swarm(6), 0.0, 6.0, _LIMITS, None))
    _all_hold(form_cone((40, 50, False, 3.0), _swarm(5), 0.0, 6.0, _LIMITS, None))
    _all_hold(center(([...],), _swarm(4), 0.0, 6.0, _LIMITS, None))
    _all_hold(line_form((300,), _swarm(5), 0.0, 6.0, _LIMITS, None))
    _all_hold(grid_form((50,), _swarm(4), 0.0, 6.0, _LIMITS, None))
    _all_hold(vee_form((45, 50), _swarm(7), 0.0, 6.0, _LIMITS, None))
    _all_hold(polygon_form((4, 100, 100), _swarm(12), 0.0, 6.0, _LIMITS, None))
    _all_hold(helix_form((60, 120, 2), _swarm(8), 0.0, 6.0, _LIMITS, None))


def test_form_circle_radius():
    out = form_circle(([...], 100, 150, 2.0), _swarm(6), 0.0, 6.0, _LIMITS, None)
    for s in out.values():
        np.testing.assert_allclose(np.linalg.norm(s.evaluate(0.0)[:2]), 100.0, atol=1e-6)


# --- Task 7: rotational transform primitives -----------------------------------------


def test_rotate_preserves_radius():
    swarm = np.array([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0]])
    out = rotate((90, "z"), swarm, 0.0, 4.0, _LIMITS, None)
    for idx, s in out.items():
        np.testing.assert_allclose(
            np.linalg.norm(s.evaluate(0.0)[:2]), np.linalg.norm(s.evaluate(2.0)[:2]), atol=1e-2
        )
        np.testing.assert_allclose(s.evaluate(0.0), swarm[idx], atol=1e-6)


def test_spiral_grows_helix_rises():
    sp = spiral((4, 150), _swarm(6), 0.0, 6.0, _LIMITS, None)
    for s in sp.values():
        assert np.linalg.norm(s.evaluate(6.0)[:2]) > np.linalg.norm(s.evaluate(0.0)[:2])
    hx = helix((4, 100, 150), _swarm(6), 0.0, 6.0, _LIMITS, None)
    for s in hx.values():
        assert s.evaluate(6.0)[2] > s.evaluate(0.0)[2]


def test_spiral_radius_follows_linear_scale_midcurve():
    # spiral computes r0 = _ring_radius_floor(60, n); the scale ramps 1 -> growth (default 2).
    from swarm_gpt.core.blocks import _ring_radius_floor

    n = 6
    growth = 2.0
    r0 = _ring_radius_floor(60.0, n)
    sp = spiral((4, 150), _swarm(n), 0.0, 6.0, _LIMITS, None)
    expected_mid = r0 * (1.0 + (growth - 1.0) * 0.5)
    for s in sp.values():
        r_start = np.linalg.norm(s.evaluate(0.0)[:2])
        r_mid = np.linalg.norm(s.evaluate(3.0)[:2])
        r_end = np.linalg.norm(s.evaluate(6.0)[:2])
        assert r_start < r_mid < r_end
        np.testing.assert_allclose(r_mid, expected_mid, atol=0.5)


def test_single_drone_does_not_blow_up():
    out = center(([...],), np.array([[0.0, 0.0, 100.0]]), 0.0, 4.0, _LIMITS, None)
    for s in out.values():
        pos = s.evaluate(2.0)
        assert np.all(np.isfinite(pos))
        assert np.all(np.abs(pos) < 1e4)


def test_orbit_translates_rigidly_and_tumble_changes_z():
    ob = orbit((90, 200), _swarm(4), 0.0, 6.0, _LIMITS, None)
    # Rigid: pairwise drone spacing is preserved through the orbit.
    p0 = np.array([s.evaluate(0.0) for s in ob.values()])
    p1 = np.array([s.evaluate(6.0) for s in ob.values()])
    np.testing.assert_allclose(
        np.linalg.norm(p0[0] - p0[1]), np.linalg.norm(p1[0] - p1[1]), atol=1.0
    )
    tb = tumble((90, "y"), _swarm(3), 0.0, 6.0, _LIMITS, None)
    assert any(abs(s.evaluate(6.0)[2] - s.evaluate(0.0)[2]) > 1.0 for s in tb.values())


def test_twister_and_move_z():
    assert len(twister((4, 20, 30), _swarm(5), 0.0, 6.0, _LIMITS, None)) == 5
    mz = move_z(([...], 50), _swarm(3), 0.0, 4.0, _LIMITS, None)
    for s in mz.values():
        np.testing.assert_allclose(s.evaluate(4.0)[2] - s.evaluate(0.0)[2], 50.0, atol=1.0)


# --- Task 8: affine transform primitives ---------------------------------------------


def test_translate_drifts_all_drones_equally():
    out = translate((100, 0, 50), _swarm(3), 0.0, 4.0, _LIMITS, None)
    for s in out.values():
        np.testing.assert_allclose(s.evaluate(4.0) - s.evaluate(0.0), [100, 0, 50], atol=1e-6)


def test_scale_expands_about_centroid():
    swarm = _swarm(4)
    c = swarm.mean(axis=0)
    out = scale((2.0,), swarm, 0.0, 4.0, _LIMITS, None)
    for s in out.values():
        r0 = np.linalg.norm(s.evaluate(0.0)[:2] - c[:2])
        r1 = np.linalg.norm(s.evaluate(4.0)[:2] - c[:2])
        np.testing.assert_allclose(r1, 2.0 * r0, atol=1e-6)


def test_shear_and_zigzag_run():
    assert len(shear((0.5, "xz"), _swarm(3), 0.0, 4.0, _LIMITS, None)) == 3
    assert len(zig_zag((4, 50, 20), _swarm(4), 0.0, 4.0, _LIMITS, None)) == 4


# --- Task 9: field primitives --------------------------------------------------------


def test_wave_is_vertical_only():
    out = wave((4, 150), _swarm(4), 0.0, 4.0, _LIMITS, None)
    for s in out.values():
        np.testing.assert_allclose(s.evaluate(0.0)[:2], s.evaluate(2.0)[:2], atol=1e-6)


def test_ripple_traveling_pulse_cascade_breathe_run():
    for out in (
        ripple((20, 1), _swarm(6), 0.0, 4.0, _LIMITS, None),
        traveling_wave((20, 1), _swarm(6), 0.0, 4.0, _LIMITS, None),
        pulse((30, 2), _swarm(6), 0.0, 4.0, _LIMITS, None),
        cascade((20, 1), _swarm(6), 0.0, 4.0, _LIMITS, None),
        breathe((1.5, 1), _swarm(6), 0.0, 4.0, _LIMITS, None),
    ):
        assert len(out) == 6


def test_cascade_phases_by_index():
    out = cascade((20, 1), _swarm(6), 0.0, 4.0, _LIMITS, None)
    # Different drones peak at different times -> their t=0 z differs.
    z0 = np.array([out[i].evaluate(0.0)[2] for i in range(6)])
    assert z0.std() > 1e-3


def test_twist_keeps_radius_per_drone():
    swarm = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 160.0]])
    out = twist((90,), swarm, 0.0, 4.0, _LIMITS, None)
    for s in out.values():
        np.testing.assert_allclose(
            np.linalg.norm(s.evaluate(0.0)[:2]), np.linalg.norm(s.evaluate(4.0)[:2]), atol=1e-2
        )


# --- move / swap (LLM-allowed targets, ported for choreographer dispatch) -------------


def test_move_translates_target_drone_to_position():
    swarm = np.array([[0.0, 0.0, 100.0], [50.0, 0.0, 100.0]])
    out = move((30.0, 40.0, 120.0, 1), swarm, 0.0, 2.0, {})  # drone_id 1 -> index 0
    assert set(out) == {0}
    np.testing.assert_allclose(out[0].evaluate(0.0), [0.0, 0.0, 100.0])
    np.testing.assert_allclose(out[0].evaluate(2.0), [30.0, 40.0, 120.0])
    assert np.any(np.abs(out[0].end_state()[1]) > 1e-6)  # real boundary velocity


def test_swap_exchanges_two_drone_positions():
    swarm = np.array([[0.0, 0.0, 100.0], [50.0, 0.0, 100.0], [99.0, 99.0, 99.0]])
    out = swap((1, 2), swarm, 0.0, 2.0, {})  # drone ids 1,2 -> indices 0,1
    assert set(out) == {0, 1}
    np.testing.assert_allclose(out[0].evaluate(0.0), swarm[0])
    np.testing.assert_allclose(out[0].evaluate(2.0), swarm[1])
    np.testing.assert_allclose(out[1].evaluate(0.0), swarm[1])
    np.testing.assert_allclose(out[1].evaluate(2.0), swarm[0])


# --- Task 10: registry + dispatch ----------------------------------------------------


def test_registry_covers_the_full_set():
    expected = {
        "form_circle",
        "form_star",
        "form_cone",
        "center",
        "line",
        "grid",
        "vee",
        "polygon",
        "helix_static",
        "rotate",
        "spiral",
        "spiral_speed",
        "helix",
        "twister",
        "orbit",
        "tumble",
        "move_z",
        "move",
        "swap",
        "zig_zag",
        "translate",
        "scale",
        "shear",
        "wave",
        "ripple",
        "traveling_wave",
        "pulse",
        "cascade",
        "breathe",
        "twist",
    }
    assert expected <= set(SPLINE_PRIMITIVES)


def test_spiral_speed_aliases_spiral_and_unknown_raises():
    assert spline_primitive_by_name("spiral_speed") is spline_primitive_by_name("spiral")
    with pytest.raises(KeyError):
        spline_primitive_by_name("teleport")

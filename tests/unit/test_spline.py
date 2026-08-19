"""Unit tests for the Bernstein spline foundation."""

import numpy as np
import pytest

from swarm_gpt.core.spline import PiecewiseSpline, Spline, SplineDict, SplinePrimitive

# ---------------------------------------------------------------------------
# Task 1: construction + evaluation
# ---------------------------------------------------------------------------


def test_linear_spline_evaluates_to_straight_line():
    # Degree-1 Bézier from (0,0,0) to (2,4,6) over [0, 1].
    spline = Spline(np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]))
    np.testing.assert_allclose(spline.evaluate(0.0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(spline.evaluate(0.5), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(spline.evaluate(1.0), [2.0, 4.0, 6.0])


def test_evaluate_accepts_array_of_times_and_respects_interval():
    spline = Spline(np.array([[0.0], [10.0]]), t0=2.0, t1=4.0)
    out = spline.evaluate(np.array([2.0, 3.0, 4.0]))
    assert out.shape == (3, 1)
    np.testing.assert_allclose(out[:, 0], [0.0, 5.0, 10.0])


def test_degree_and_duration_properties():
    spline = Spline(np.zeros((4, 3)), t0=1.0, t1=3.5)
    assert spline.degree == 3
    assert spline.dim == 3
    assert spline.duration == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Task 2: derivative
# ---------------------------------------------------------------------------


def test_derivative_of_line_is_constant_velocity():
    # Line covering 6 m over 3 s -> constant 2 m/s on each axis it moves.
    spline = Spline(np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]), t0=0.0, t1=3.0)
    vel = spline.derivative()
    assert vel.degree == 0
    np.testing.assert_allclose(vel.evaluate(0.0), [2.0, 0.0, 0.0])
    np.testing.assert_allclose(vel.evaluate(3.0), [2.0, 0.0, 0.0])


def test_derivative_matches_finite_difference():
    rng = np.random.default_rng(0)
    spline = Spline(rng.standard_normal((5, 3)), t0=0.0, t1=2.0)
    deriv = spline.derivative()
    t, h = 0.7, 1e-6
    fd = (spline.evaluate(t + h) - spline.evaluate(t - h)) / (2 * h)
    np.testing.assert_allclose(deriv.evaluate(t), fd, atol=1e-4)


# ---------------------------------------------------------------------------
# Task 3: endpoint states
# ---------------------------------------------------------------------------


def test_endpoint_states_read_off_control_points():
    # Degree-3 spline; states must match the chained derivatives exactly.
    rng = np.random.default_rng(1)
    spline = Spline(rng.standard_normal((4, 3)), t0=0.0, t1=1.5)
    v = spline.derivative()
    a = v.derivative()

    p0, v0, a0 = spline.start_state()
    np.testing.assert_allclose(p0, spline.evaluate(0.0))
    np.testing.assert_allclose(v0, v.evaluate(0.0))
    np.testing.assert_allclose(a0, a.evaluate(0.0))

    p1, v1, a1 = spline.end_state()
    np.testing.assert_allclose(p1, spline.evaluate(1.5))
    np.testing.assert_allclose(v1, v.evaluate(1.5))
    np.testing.assert_allclose(a1, a.evaluate(1.5))


# ---------------------------------------------------------------------------
# Task 4: degree elevation
# ---------------------------------------------------------------------------


def test_degree_elevation_preserves_the_curve():
    rng = np.random.default_rng(2)
    spline = Spline(rng.standard_normal((3, 3)), t0=0.0, t1=2.0)
    elevated = spline.degree_elevate(6)
    assert elevated.degree == 6
    for t in np.linspace(0.0, 2.0, 11):
        np.testing.assert_allclose(elevated.evaluate(t), spline.evaluate(t), atol=1e-12)


def test_degree_elevation_to_same_degree_is_noop():
    spline = Spline(np.array([[0.0], [1.0], [2.0]]))
    assert spline.degree_elevate(2).degree == 2


# ---------------------------------------------------------------------------
# Task 5: addition
# ---------------------------------------------------------------------------


def test_addition_is_pointwise_and_handles_mixed_degree():
    # A degree-1 spline plus a degree-2 spline, same interval.
    a = Spline(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), t0=0.0, t1=1.0)
    b = Spline(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), t0=0.0, t1=1.0)
    s = a + b
    assert s.degree == 2
    for t in np.linspace(0.0, 1.0, 9):
        np.testing.assert_allclose(s.evaluate(t), a.evaluate(t) + b.evaluate(t), atol=1e-12)


def test_addition_rejects_mismatched_intervals():
    a = Spline(np.zeros((2, 3)), t0=0.0, t1=1.0)
    b = Spline(np.zeros((2, 3)), t0=0.0, t1=2.0)
    with pytest.raises(ValueError, match="interval"):
        _ = a + b


# ---------------------------------------------------------------------------
# Task 6: multiplication
# ---------------------------------------------------------------------------


def test_scalar_float_multiplication():
    spline = Spline(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    scaled = spline * 2.0
    np.testing.assert_allclose(scaled.control_points, spline.control_points * 2.0)
    # __rmul__ path
    np.testing.assert_allclose((2.0 * spline).control_points, spline.control_points * 2.0)


def test_scalar_spline_times_scalar_spline_is_pointwise_product():
    # p(u) linear 0->2, q(u) linear 1->3, both over [0,1]; product is degree 2.
    p = Spline(np.array([[0.0], [2.0]]))
    q = Spline(np.array([[1.0], [3.0]]))
    prod = p * q
    assert prod.degree == 2
    for t in np.linspace(0.0, 1.0, 9):
        np.testing.assert_allclose(prod.evaluate(t), p.evaluate(t) * q.evaluate(t), atol=1e-12)


def test_scalar_spline_times_vector_spline_scales_each_axis():
    scale = Spline(np.array([[1.0], [3.0]]))  # 1 -> 3 over [0,1]
    curve = Spline(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))  # constant (1,1,1)
    scaled = scale * curve
    assert scaled.dim == 3
    for t in np.linspace(0.0, 1.0, 5):
        expected = scale.evaluate(t) * curve.evaluate(t)
        np.testing.assert_allclose(scaled.evaluate(t), expected, atol=1e-12)


def test_vector_times_vector_is_rejected():
    a = Spline(np.zeros((2, 3)))
    b = Spline(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="scalar"):
        _ = a * b


# ---------------------------------------------------------------------------
# Task 7: affine transform
# ---------------------------------------------------------------------------


def test_affine_transform_commutes_with_evaluation():
    rng = np.random.default_rng(3)
    spline = Spline(rng.standard_normal((4, 3)), t0=0.0, t1=1.0)
    matrix = np.array(
        [[0.0, -1.0, 0.0, 5.0], [1.0, 0.0, 0.0, -2.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    )
    placed = spline.affine_transform(matrix)
    for t in np.linspace(0.0, 1.0, 7):
        point = spline.evaluate(t)
        expected = (matrix @ np.append(point, 1.0))[:3]
        np.testing.assert_allclose(placed.evaluate(t), expected, atol=1e-12)


def test_affine_transform_requires_3d():
    scalar = Spline(np.zeros((2, 1)))
    with pytest.raises(ValueError, match="3-D"):
        scalar.affine_transform(np.eye(4))


def test_affine_transform_accepts_list_of_lists():
    spline = Spline(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    matrix_as_list = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    result = spline.affine_transform(matrix_as_list)
    np.testing.assert_allclose(result.control_points, spline.control_points)


def test_affine_transform_wrong_shape_raises_value_error():
    spline = Spline(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        spline.affine_transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# Task 8: axis bounds
# ---------------------------------------------------------------------------


def test_axis_bounds_contain_the_sampled_curve():
    rng = np.random.default_rng(4)
    spline = Spline(rng.standard_normal((6, 3)), t0=0.0, t1=1.0)
    lower, upper = spline.axis_bounds()
    samples = spline.evaluate(np.linspace(0.0, 1.0, 200))
    assert np.all(samples >= lower - 1e-12)
    assert np.all(samples <= upper + 1e-12)


# ---------------------------------------------------------------------------
# Task 9: to_waypoints
# ---------------------------------------------------------------------------


def test_to_waypoints_samples_at_frequency():
    spline = Spline(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), t0=0.0, t1=1.0)
    wp = spline.to_waypoints(freq=10.0)
    assert wp["time"].shape == (11,)
    assert wp["pos"].shape == (11, 3)
    assert wp["vel"].shape == (11, 3)
    assert wp["acc"].shape == (11, 3)
    np.testing.assert_allclose(wp["time"][[0, -1]], [0.0, 1.0])
    np.testing.assert_allclose(wp["pos"][-1], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(wp["vel"][0], [10.0, 0.0, 0.0])  # constant 10 m/s
    np.testing.assert_allclose(wp["acc"][0], [0.0, 0.0, 0.0])


def test_to_waypoints_non_integer_duration_preserves_endpoint():
    # duration * freq = 1.04 * 10 = 10.4, non-integer: old arange-based grid ended at t=1.0.
    spline = Spline(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), t0=0.0, t1=1.04)
    wp = spline.to_waypoints(freq=10.0)
    np.testing.assert_allclose(wp["time"][-1], 1.04)
    np.testing.assert_allclose(wp["pos"][-1], [10.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Task 10: PiecewiseSpline
# ---------------------------------------------------------------------------


def test_piecewise_dispatches_by_interval():
    seg_a = Spline(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), t0=0.0, t1=1.0)
    seg_b = Spline(np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]), t0=1.0, t1=2.0)
    pw = PiecewiseSpline([seg_a, seg_b])
    np.testing.assert_allclose(pw.evaluate(0.5), [0.5, 0.0, 0.0])
    np.testing.assert_allclose(pw.evaluate(1.0), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(pw.evaluate(1.5), [1.0, 1.0, 0.0])


def test_piecewise_rejects_non_contiguous_segments():
    seg_a = Spline(np.zeros((2, 3)), t0=0.0, t1=1.0)
    seg_b = Spline(np.zeros((2, 3)), t0=1.5, t1=2.0)
    with pytest.raises(ValueError, match="contiguous"):
        PiecewiseSpline([seg_a, seg_b])


def test_piecewise_endpoint_states_span_the_whole_curve():
    seg_a = Spline(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), t0=0.0, t1=1.0)
    seg_b = Spline(np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]), t0=1.0, t1=2.0)
    pw = PiecewiseSpline([seg_a, seg_b])
    np.testing.assert_allclose(pw.start_state()[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(pw.end_state()[0], [1.0, 2.0, 0.0])


# ---------------------------------------------------------------------------
# Task 11: axswarm interop
# ---------------------------------------------------------------------------


def test_from_axswarm_zeta_round_trips_layout():
    # zeta = [x_0,x_1, y_0,y_1, z_0,z_1] for a degree-1 (N=1) curve.
    zeta = np.array([0.0, 2.0, 0.0, 4.0, 0.0, 6.0])
    spline = Spline.from_axswarm_zeta(zeta, degree=1, t0=0.0, t1=1.0)
    np.testing.assert_allclose(spline.control_points, [[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])


def test_spline_matches_axswarm_bernstein_basis():
    axswarm_spline = pytest.importorskip("axswarm.spline")
    k_steps, degree, freq = 20, 5, 10
    w_matrix, _, _ = axswarm_spline.bernstein_matrices(k_steps, degree, freq)
    w_matrix = np.asarray(w_matrix)

    rng = np.random.default_rng(5)
    zeta = rng.standard_normal(3 * (degree + 1))
    axswarm_positions = (w_matrix @ zeta).reshape(k_steps, 3)

    duration = (k_steps - 1) / freq
    spline = Spline.from_axswarm_zeta(zeta, degree=degree, t0=0.0, t1=duration)
    sample_times = np.arange(k_steps) / freq
    # axswarm computes bernstein_matrices in JAX float32; atol=1e-5 accommodates float32 precision.
    # The layout (zeta = [x_0..x_N, y_0..y_N, z_0..z_N]) is confirmed correct by this agreement.
    np.testing.assert_allclose(spline.evaluate(sample_times), axswarm_positions, atol=1e-5)


# ---------------------------------------------------------------------------
# Task 12: SplineDict + SplinePrimitive contract
# ---------------------------------------------------------------------------


def test_a_toy_primitive_satisfies_the_spline_contract():
    def hover(duration: float) -> SplineDict:
        cp = np.zeros((2, 3))
        return {0: Spline(cp, 0.0, duration)}

    # Structural typing: the toy primitive is a valid SplinePrimitive.
    primitive: SplinePrimitive = hover
    result = primitive(2.0)
    assert set(result) == {0}
    assert isinstance(result[0], Spline)
    assert result[0].duration == 2.0


# ---------------------------------------------------------------------------
# Clamping contract
# ---------------------------------------------------------------------------


def test_spline_evaluate_clamps_above_t1():
    spline = Spline(np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]), t0=0.0, t1=1.0)
    np.testing.assert_allclose(spline.evaluate(2.0), [5.0, 5.0, 5.0])


def test_piecewise_evaluate_clamps_below_t0():
    seg = Spline(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), t0=1.0, t1=2.0)
    pw = PiecewiseSpline([seg])
    np.testing.assert_allclose(pw.evaluate(0.0), [1.0, 2.0, 3.0])


def test_piecewise_evaluate_clamps_above_t1():
    seg = Spline(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), t0=1.0, t1=2.0)
    pw = PiecewiseSpline([seg])
    np.testing.assert_allclose(pw.evaluate(5.0), [4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# 0-d array input to evaluate
# ---------------------------------------------------------------------------


def test_evaluate_zero_d_array_returns_shape_dim():
    spline = Spline(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    result = spline.evaluate(np.array(0.5))
    assert result.shape == (3,)


# ---------------------------------------------------------------------------
# PiecewiseSpline derivative correctness
# ---------------------------------------------------------------------------


def test_piecewise_derivative_matches_segment_analytic_derivative():
    rng = np.random.default_rng(7)
    seg_a = Spline(rng.standard_normal((4, 3)), t0=0.0, t1=1.0)
    seg_b = Spline(rng.standard_normal((4, 3)), t0=1.0, t1=2.0)
    # Ensure continuity in position only (velocity discontinuity is fine for this test).
    pw = PiecewiseSpline([seg_a, seg_b])
    pw_deriv = pw.derivative()

    # Sample inside seg_a
    for t in np.linspace(0.1, 0.9, 5):
        np.testing.assert_allclose(pw_deriv.evaluate(t), seg_a.derivative().evaluate(t), atol=1e-12)
    # Sample inside seg_b
    for t in np.linspace(1.1, 1.9, 5):
        np.testing.assert_allclose(pw_deriv.evaluate(t), seg_b.derivative().evaluate(t), atol=1e-12)


# ---------------------------------------------------------------------------
# High-degree evaluate stability (De Casteljau reference)
# ---------------------------------------------------------------------------


def _de_casteljau(cp: np.ndarray, u: float) -> np.ndarray:
    """Evaluate a Bézier curve at parameter u in [0,1] via De Casteljau."""
    pts = cp.copy()
    n = len(pts) - 1
    for r in range(1, n + 1):
        pts[: n - r + 1] = (1.0 - u) * pts[: n - r + 1] + u * pts[1 : n - r + 2]
    return pts[0]


def test_high_degree_evaluate_matches_de_casteljau():
    rng = np.random.default_rng(8)
    degree = 16
    cp = rng.standard_normal((degree + 1, 3))
    spline = Spline(cp, t0=0.0, t1=1.0)
    for t in np.linspace(0.1, 0.9, 7):
        expected = _de_casteljau(cp, t)
        np.testing.assert_allclose(spline.evaluate(t), expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Subdivision (de Casteljau split) — WS2
# ---------------------------------------------------------------------------


def test_subdivide_halves_reproduce_the_original_curve():
    rng = np.random.default_rng(9)
    spline = Spline(rng.standard_normal((5, 3)), t0=1.0, t1=4.0)
    left, right = spline.subdivide(2.5)
    assert (left.t0, left.t1) == (1.0, 2.5)
    assert (right.t0, right.t1) == (2.5, 4.0)
    for t in np.linspace(1.0, 2.5, 9):
        np.testing.assert_allclose(left.evaluate(t), spline.evaluate(t), atol=1e-10)
    for t in np.linspace(2.5, 4.0, 9):
        np.testing.assert_allclose(right.evaluate(t), spline.evaluate(t), atol=1e-10)


def test_subdivide_endpoints_and_join_match():
    spline = Spline(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), t0=0.0, t1=3.0)
    left, right = spline.subdivide(1.0)
    np.testing.assert_allclose(left.evaluate(0.0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(left.evaluate(1.0), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(right.evaluate(1.0), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(right.evaluate(3.0), [3.0, 0.0, 0.0])


def test_subdivide_rejects_time_outside_interval():
    spline = Spline(np.zeros((3, 3)), t0=0.0, t1=1.0)
    with pytest.raises(ValueError, match="inside"):
        spline.subdivide(1.0)


def test_piecewise_subdivide_splits_inside_a_segment():
    seg_a = Spline(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), t0=0.0, t1=1.0)
    seg_b = Spline(np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]), t0=1.0, t1=2.0)
    pw = PiecewiseSpline([seg_a, seg_b])
    left, right = pw.subdivide(1.5)
    assert left.t0 == 0.0 and left.t1 == 1.5
    assert right.t0 == 1.5 and right.t1 == 2.0
    for t in np.linspace(0.0, 1.5, 9):
        np.testing.assert_allclose(left.evaluate(t), pw.evaluate(t), atol=1e-10)
    for t in np.linspace(1.5, 2.0, 9):
        np.testing.assert_allclose(right.evaluate(t), pw.evaluate(t), atol=1e-10)


def test_piecewise_subdivide_at_a_breakpoint_is_clean():
    seg_a = Spline(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), t0=0.0, t1=1.0)
    seg_b = Spline(np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]), t0=1.0, t1=2.0)
    pw = PiecewiseSpline([seg_a, seg_b])
    left, right = pw.subdivide(1.0)
    assert len(left.segments) == 1 and len(right.segments) == 1
    np.testing.assert_allclose(left.end_state()[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(right.start_state()[0], [1.0, 0.0, 0.0])

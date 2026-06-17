"""Unit tests for M(t) transform builders (WS1)."""

from __future__ import annotations

import numpy as np

from swarm_gpt.core.spline import PiecewiseSpline, Spline
from swarm_gpt.core.transforms import CANONICAL_ARC, arc_spline


def test_canonical_arc_radial_error_is_pinned() -> None:
    """CANONICAL_ARC has < 0.03 % radial error on the unit quarter-circle."""
    arc = Spline(np.hstack([CANONICAL_ARC, np.zeros((4, 1))]), t0=0.0, t1=1.0)
    radii = np.linalg.norm(arc.evaluate(np.linspace(0.0, 1.0, 200))[:, :2], axis=1)
    assert float(np.max(np.abs(radii - 1.0))) < 3e-4


def test_arc_spline_full_circle_keeps_radius() -> None:
    """arc_spline for a full 2π sweep stays within 1 mm of the target radius."""
    arc = arc_spline(np.array([1.0, 1.0, 0.5]), 2.0, 0.0, 2 * np.pi, 0.0, 4.0)
    assert isinstance(arc, PiecewiseSpline)
    np.testing.assert_allclose(arc.evaluate(0.0), [3.0, 1.0, 0.5], atol=1e-9)
    samples = arc.evaluate(np.linspace(0.0, 4.0, 50))
    radii = np.linalg.norm(samples[:, :2] - np.array([1.0, 1.0]), axis=1)
    np.testing.assert_allclose(radii, 2.0, atol=1e-3)


def test_arc_spline_quarter_sweep_endpoints() -> None:
    """arc_spline for a quarter circle starts at (1,0,0) and ends at (0,1,0)."""
    arc = arc_spline(np.zeros(3), 1.0, 0.0, np.pi / 2, 0.0, 1.0)
    np.testing.assert_allclose(arc.evaluate(0.0), [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(arc.evaluate(1.0), [0.0, 1.0, 0.0], atol=1e-3)


def test_arc_spline_negative_sweep() -> None:
    """arc_spline for a -π/2 sweep starts at (1,0,0), ends near (0,-1,0), radius preserved."""
    arc = arc_spline(np.zeros(3), 1.0, 0.0, -np.pi / 2, 0.0, 1.0)
    np.testing.assert_allclose(arc.evaluate(0.0), [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(arc.evaluate(1.0), [0.0, -1.0, 0.0], atol=1e-3)
    samples = arc.evaluate(np.linspace(0.0, 1.0, 50))
    radii = np.linalg.norm(samples[:, :2], axis=1)
    np.testing.assert_allclose(radii, 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Task 3: Polynomial transforms
# ---------------------------------------------------------------------------
from swarm_gpt.core.transforms import (  # noqa: E402
    affine_offset,
    linear_scale,
    linear_translate,
    zigzag_translate,
)


def test_linear_translate_and_scale() -> None:
    """linear_translate reaches end point; linear_scale ramps scalar correctly."""
    tr = linear_translate(np.zeros(3), np.array([6.0, 0.0, 3.0]), 0.0, 3.0)
    np.testing.assert_allclose(tr.evaluate(3.0), [6.0, 0.0, 3.0])
    sc = linear_scale(1.0, 3.0, 0.0, 2.0)
    assert sc.dim == 1
    np.testing.assert_allclose(sc.evaluate(2.0), [3.0])


def test_affine_offset_scales_about_center() -> None:
    """affine_offset: displacement is zero at t0 and equals matrix_end@offset−offset at t1."""
    # Scale a point at (2,0,0) about origin by factor 3 over the block -> ends at (6,0,0).
    home = np.array([2.0, 0.0, 0.0])
    matrix_end = np.diag([3.0, 3.0, 1.0])
    off = affine_offset(home, np.zeros(3), matrix_end, 0.0, 4.0)
    np.testing.assert_allclose(off.evaluate(0.0), [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(off.evaluate(4.0), [4.0, 0.0, 0.0], atol=1e-9)  # +4 displacement


def test_zigzag_alternates_direction() -> None:
    """zigzag_translate alternates horizontal direction while accumulating z each step."""
    zz = zigzag_translate(
        np.array([0.0, 0.0, 1.0]), 4, np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.5]), 0.0, 4.0
    )
    np.testing.assert_allclose(zz.evaluate(1.0), [1.0, 1.0, 1.5], atol=1e-9)
    np.testing.assert_allclose(zz.evaluate(2.0), [0.0, 0.0, 2.0], atol=1e-9)

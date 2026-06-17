"""Unit tests for formation (R) generators (WS1)."""

import numpy as np

from swarm_gpt.core.formations import cone, grid, helix_static, line, polygon, ring, star, vee


def test_ring_radius_count_offset():
    pts = ring(6, 100.0, 150.0)
    np.testing.assert_allclose(np.linalg.norm(pts[:, :2], axis=1), 100.0)
    np.testing.assert_allclose(pts[:, 2], 150.0)
    off = ring(4, 50.0, 0.0, angle_offset=np.pi / 2)
    np.testing.assert_allclose(off[0, :2], ring(4, 50.0, 0.0)[1, :2], atol=1e-9)


def test_grid_centered():
    pts = grid(4, 50.0, 100.0, np.array([10.0, 20.0]))
    np.testing.assert_allclose(pts[:, :2].mean(axis=0), [10.0, 20.0], atol=1e-9)


def test_line_endpoints():
    pts = line(5, np.array([0.0, 0.0, 100.0]), np.array([400.0, 0.0, 100.0]))
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 100.0])
    np.testing.assert_allclose(pts[-1], [400.0, 0.0, 100.0])


def test_vee_is_symmetric_two_arms():
    pts = vee(7, np.array([0.0, 0.0, 120.0]), spread=np.deg2rad(45), spacing=50.0)
    assert pts.shape == (7, 3)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 120.0])  # apex
    np.testing.assert_allclose(pts[:, 0].max(), 0.0, atol=1e-9)  # arms go backward (-x)


def test_polygon_vertices_on_circumradius():
    pts = polygon(12, n_sides=4, radius=100.0, height=100.0)
    assert pts.shape == (12, 3)
    assert np.linalg.norm(pts[:, :2], axis=1).max() <= 100.0 + 1e-6


def test_star_two_radii_and_cone_layers():
    s = star(6, 80.0, 40.0, 120.0)
    assert set(np.round(np.linalg.norm(s[:, :2], axis=1), 3)) == {80.0, 120.0}
    c = cone(5, 50.0, 200.0, -30.0)
    assert c[0, 2] == 200.0 and c[1:, 2].max() <= 200.0


def test_helix_static_rises_and_circles():
    h = helix_static(8, radius=60.0, z0=100.0, pitch=120.0, turns=2.0)
    assert h.shape == (8, 3)
    assert h[-1, 2] > h[0, 2]
    np.testing.assert_allclose(np.linalg.norm(h[:, :2], axis=1), 60.0)

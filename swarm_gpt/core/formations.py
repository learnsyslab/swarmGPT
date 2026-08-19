"""Formation (R) generators for WS1: fixed home positions in cm, shape (n, 3).

Drone-to-home assignment is the caller's responsibility (Hungarian + min-snap). Layout
math is harvested from the legacy primitives plus new line/grid/vee/polygon/helix layouts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def ring(n: int, radius: float, height: float, angle_offset: float = 0.0) -> NDArray:
    """Return ``n`` equally spaced points on a horizontal circle (cm).

    Args:
        n: Number of drones.
        radius: Circle radius in cm.
        height: Height (z) of all points in cm.
        angle_offset: Starting angle offset in radians.

    Returns:
        Home positions of shape ``(n, 3)`` in cm.
    """
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False) + angle_offset
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(n, height)], axis=1)


def grid(n: int, spacing: float, height: float, center: NDArray) -> NDArray:
    """Return ``n`` points on a centered rectangular grid (cm).

    Args:
        n: Number of drones.
        spacing: Grid spacing in cm.
        height: Height (z) of all points in cm.
        center: 2-D center ``(cx, cy)`` of the grid in cm.

    Returns:
        Home positions of shape ``(n, 3)`` in cm.
    """
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    gx, gy = np.meshgrid(np.arange(cols) * spacing, np.arange(rows) * spacing)
    x = (gx.flatten() - gx.mean())[:n] + center[0]
    y = (gy.flatten() - gy.mean())[:n] + center[1]
    return np.stack([x, y, np.full(n, height)], axis=1)


def line(n: int, start: NDArray, end: NDArray) -> NDArray:
    """Return ``n`` points evenly spaced from ``start`` to ``end`` (cm).

    Args:
        n: Number of drones.
        start: Start position ``(x, y, z)`` in cm.
        end: End position ``(x, y, z)`` in cm.

    Returns:
        Home positions of shape ``(n, 3)`` in cm.
    """
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - t) * start[None, :] + t * end[None, :]


def vee(n: int, apex: NDArray, spread: float, spacing: float) -> NDArray:
    """Return a V / flying-wedge: an apex with two backward arms (cm).

    Args:
        n: Total drones (apex + two arms).
        apex: Apex position ``(x, y, z)`` in cm.
        spread: Half-angle of the V in radians (measured from the -x axis).
        spacing: Spacing between consecutive drones along each arm, in cm.

    Returns:
        Home positions of shape ``(n, 3)`` in cm with the apex first.
    """
    pts = [apex]
    per_arm = (n - 1) // 2
    for arm_sign in (1.0, -1.0):
        direction = np.array([-np.cos(spread), arm_sign * np.sin(spread), 0.0])
        for k in range(1, per_arm + 1):
            pts.append(apex + direction * spacing * k)
    if len(pts) < n:  # odd leftover -> extend the first arm
        direction = np.array([-np.cos(spread), np.sin(spread), 0.0])
        pts.append(apex + direction * spacing * (per_arm + 1))
    return np.array(pts[:n])


def polygon(n: int, n_sides: int, radius: float, height: float) -> NDArray:
    """Return ``n`` points distributed along a regular ``n_sides``-gon perimeter (cm).

    Args:
        n: Number of drones.
        n_sides: Number of polygon sides.
        radius: Circumradius of the polygon in cm.
        height: Height (z) of all points in cm.

    Returns:
        Home positions of shape ``(n, 3)`` in cm.
    """
    verts = ring(n_sides, radius, height)
    per_edge = np.full(n_sides, n // n_sides)
    per_edge[: n % n_sides] += 1
    pts = []
    for s in range(n_sides):
        a, b = verts[s], verts[(s + 1) % n_sides]
        for k in range(per_edge[s]):
            pts.append(a + (b - a) * (k / per_edge[s]))
    return np.array(pts[:n])


def star(n: int, radius: float, delta_radius: float, height: float) -> NDArray:
    """Return a star: two offset rings plus an optional center point (cm).

    Args:
        n: Number of drones.
        radius: Inner ring radius in cm.
        delta_radius: Additional radius for the outer ring in cm.
        height: Height (z) of all points in cm.

    Returns:
        Home positions of shape ``(n, 3)`` in cm.
    """
    per = n // 2
    pts = np.vstack(
        [
            ring(per, radius, height),
            ring(per, radius + delta_radius, height, angle_offset=np.pi / per if per else 0.0),
        ]
    )
    if n != per * 2:
        pts = np.vstack([pts, np.array([0.0, 0.0, height])])
    return pts


def cone_layers(n: int, layer_growth: int = 4) -> int:
    """Number of rings :func:`cone` stacks below its apex, so a caller can bound the z span.

    Args:
        n: Number of drones.
        layer_growth: Number of additional drones per layer.

    Returns:
        The ring count (zero for a lone apex).
    """
    left, in_layer, layers = n - 1, 0, 0
    while left > 0:
        in_layer += layer_growth
        left -= min(in_layer, left)
        layers += 1
    return layers


def cone(n: int, spacing: float, z0: float, delta_h: float, layer_growth: int = 4) -> NDArray:
    """Return a cone: an apex with widening z-stacked rings (cm).

    Args:
        n: Number of drones.
        spacing: Approximate inter-drone spacing in cm (controls ring radius).
        z0: Height of the apex in cm.
        delta_h: Height step per layer in cm (negative goes downward).
        layer_growth: Number of additional drones per layer (default 4).

    Returns:
        Home positions of shape ``(n, 3)`` in cm with the apex first.
    """
    pts = [np.array([0.0, 0.0, z0])]
    left, in_layer, z = n - 1, 0, z0
    while left > 0:
        in_layer += layer_growth
        z += delta_h
        count = min(in_layer, left)
        pts.append(ring(count, spacing / (2 * np.sin(np.pi / in_layer)), z))
        left -= count
    return np.vstack(pts)


def helix_static(n: int, radius: float, z0: float, pitch: float, turns: float) -> NDArray:
    """Return ``n`` points frozen along a helix of ``turns`` turns and total rise (cm).

    Args:
        n: Number of drones.
        radius: Helix radius in cm.
        z0: Starting height in cm.
        pitch: Height gain per turn in cm.
        turns: Number of full turns.

    Returns:
        Home positions of shape ``(n, 3)`` in cm.
    """
    # Exclusive in angle: closing the turn would put the last drone directly above the first,
    # separated only by the rise.
    a = np.linspace(0.0, 2 * np.pi * turns, n, endpoint=False)
    z = z0 + np.linspace(0.0, pitch * turns, n)
    return np.stack([radius * np.cos(a), radius * np.sin(a), z], axis=1)

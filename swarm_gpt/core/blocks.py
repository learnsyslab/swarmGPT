"""Block composition evaluator and primitive registry for WS1.

Evaluates ``p_i(t) = M(t) * (R_i + sum_k D_k(R_i, t))`` into one ``Spline`` (or
``PiecewiseSpline``) per drone (spline 1). Each primitive is an ``(R, [D], M)`` triple.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from swarm_gpt.core import fields, formations
from swarm_gpt.core.motion_primitives import _assign_positions, _sanitize_drone_ids
from swarm_gpt.core.spline import PiecewiseSpline, Spline
from swarm_gpt.core.transforms import (
    affine_offset,
    arc_spline,
    linear_scale,
    linear_translate,
    zigzag_translate,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from swarm_gpt.core.spline import SplineDict

_Curve = Spline | PiecewiseSpline


def constant_spline(point: NDArray, t0: float, t1: float) -> Spline:
    """Lift a fixed point (cm) to a degree-0 constant spline over ``[t0, t1]``.

    Args:
        point: Fixed position ``(x, y, z)`` in cm.
        t0: Interval start time in seconds.
        t1: Interval end time in seconds.

    Returns:
        A degree-0 ``Spline`` holding ``point`` over ``[t0, t1]``.
    """
    return Spline(np.asarray(point, dtype=float)[None, :], t0=t0, t1=t1)


def _ring_radius_floor(spacing: float, n: int) -> float:
    """Minimum-spacing ring radius in cm; 0 for a single drone (avoids div-by-zero).

    Args:
        spacing: Desired minimum chord spacing between adjacent drones in cm.
        n: Number of drones on the full ring.

    Returns:
        The ring radius giving ``spacing`` between neighbours, or 0 for ``n < 2``.
    """
    return 0.0 if n < 2 else spacing / (2.0 * np.sin(np.pi / n))


def _add_constant(curve: _Curve, point: NDArray) -> _Curve:
    """Add a constant 3-vector ``point`` (cm) to a curve over its own interval(s).

    Args:
        curve: A ``Spline`` or ``PiecewiseSpline`` displacement.
        point: A constant ``(x, y, z)`` offset in cm (e.g. the drone's home).

    Returns:
        The summed curve, matching the input's segmentation.
    """
    if isinstance(curve, Spline):
        return curve + constant_spline(point, curve.t0, curve.t1)
    return PiecewiseSpline([seg + constant_spline(point, seg.t0, seg.t1) for seg in curve.segments])


def _add_curves(a: _Curve, b: _Curve) -> _Curve:
    """Add two displacement curves over the same ``[t0, t1]``.

    Args:
        a: A ``Spline`` or ``PiecewiseSpline`` displacement.
        b: A second displacement over the same interval and segmentation.

    Returns:
        The summed curve.

    Raises:
        NotImplementedError: If the two curves have mismatched segmentation.
    """
    if isinstance(a, Spline) and isinstance(b, Spline):
        return a + b
    if isinstance(a, PiecewiseSpline) and isinstance(b, PiecewiseSpline):
        a_breaks = [(seg.t0, seg.t1) for seg in a.segments]
        b_breaks = [(seg.t0, seg.t1) for seg in b.segments]
        if a_breaks == b_breaks:
            return PiecewiseSpline([sa + sb for sa, sb in zip(a.segments, b.segments)])
    raise NotImplementedError("cannot add curves with mismatched segmentation")


def compose_block(
    homes: dict[int, NDArray],
    field_layers: list[dict[int, _Curve]],
    transforms: dict[int, Callable[[Spline], _Curve]] | None,
    t0: float,
    t1: float,
) -> SplineDict:
    """Compose ``M(t) * (R_i + sum_k D_k)`` into one curve per drone.

    Args:
        homes: Drone id -> home position ``(x, y, z)`` in cm (R).
        field_layers: List of per-drone displacement curves to sum (D layers).
        transforms: Optional per-drone callable applying ``M(t)`` to the home spline.
        t0: Block start time in seconds.
        t1: Block end time in seconds.

    Returns:
        Drone id -> composed spline-1 trajectory.
    """
    out: SplineDict = {}
    for idx, home in homes.items():
        disp: _Curve | None = None
        for layer in field_layers:
            if idx in layer:
                disp = layer[idx] if disp is None else _add_curves(disp, layer[idx])
        base: _Curve = constant_spline(home, t0, t1) if disp is None else _add_constant(disp, home)
        if transforms is not None and idx in transforms:
            base = transforms[idx](base)
        out[idx] = base
    return out


def _hold(homes: NDArray, ids: list[int], t0: float, t1: float) -> SplineDict:
    """Return a constant-hold spline per assigned drone.

    Args:
        homes: Assigned home positions in cm, shape ``(n, 3)``.
        ids: Drone ids in the same row order as ``homes``.
        t0: Block start time in seconds.
        t1: Block end time in seconds.

    Returns:
        Drone id -> constant-hold spline.
    """
    return {d: constant_spline(homes[row], t0, t1) for row, d in enumerate(ids)}


def _assign(
    swarm_pos: NDArray, homes: NDArray, ids: list[int], vel: NDArray | None, t: float
) -> NDArray:
    """Assign ``ids`` rows of ``swarm_pos`` to ``homes`` and return assigned homes.

    Args:
        swarm_pos: Current swarm positions in cm, shape ``(n, 3)``.
        homes: Candidate target positions in cm.
        ids: Drone ids (rows of ``swarm_pos``) to assign.
        vel: Optional per-drone velocities in cm/s, shape ``(n, 3)``.
        t: Time horizon in seconds for the assignment cost.

    Returns:
        The assigned home positions in cm, in ``ids`` row order.
    """
    v = vel[ids] if vel is not None else None
    return homes[_assign_positions(swarm_pos[ids], homes, swarm_vel=v, T=t)]


def _assign_to_motion(
    swarm_pos: NDArray,
    swarm_vel: NDArray | None,
    layout: NDArray,
    ids: list[int],
    t0: float,
    t1: float,
    build: Callable[[NDArray], SplineDict],
) -> SplineDict:
    """Assign drones to moving-primitive slots, accounting for each slot's arrival velocity.

    The motion ``build`` is evaluated once on the geometric ``layout`` (slot order), giving each
    slot's curve and its start velocity ``v_f``. Drones are then assigned to slots with a
    velocity-aware min-snap cost that includes ``v_f`` (so a drone already moving toward a slot's
    motion is cheaper to place there), and each drone receives its assigned slot's curve.

    Args:
        swarm_pos: Current swarm positions in cm, shape ``(n, 3)``.
        swarm_vel: Current per-drone velocities in cm/s, or ``None``.
        layout: Slot start positions in cm, shape ``(m, 3)``.
        ids: Drone ids to assign (rows of ``swarm_pos``).
        t0: Block start time in seconds.
        t1: Block end time in seconds.
        build: Builds the per-slot motion curves from start positions; returns ``{row: curve}``.

    Returns:
        Drone id -> assigned motion curve.
    """
    slot_curves = build(layout)
    vf = np.array([np.asarray(slot_curves[j].start_state()[1]) for j in range(len(layout))])
    v0 = swarm_vel[ids] if swarm_vel is not None else None
    perm = _assign_positions(swarm_pos[ids], layout, swarm_vel=v0, T=t1 - t0, target_vel=vf)
    return {ids[i]: slot_curves[int(perm[i])] for i in range(len(ids))}


def form_circle(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Ring formation held on a circle of the given radius/height."""
    drone_ids, radius_cm, z_cm, _t = params
    ids = _sanitize_drone_ids(drone_ids, swarm_pos.shape[0])
    n = len(ids)
    r = max(float(radius_cm), _ring_radius_floor(80.0, n))
    homes = _assign(swarm_pos, formations.ring(n, r, float(z_cm)), ids, swarm_vel, tend - tstart)
    return _hold(homes, ids, tstart, tend)


def form_star(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Star formation (two offset rings + optional center), held."""
    height, min_spacing, delta_radius, _t = params
    n = swarm_pos.shape[0]
    per = n // 2
    r = max(min_spacing, 40.0) / (2 * np.sin(np.pi / per))
    homes = formations.star(n, r, max(delta_radius, 40.0), int(height))
    ids = list(range(n))
    return _hold(_assign(swarm_pos, homes, ids, swarm_vel, tend - tstart), ids, tstart, tend)


def form_cone(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Cone formation (apex + widening z-rings), held."""
    delta_height, spacing, is_inverted, _t = params
    n = swarm_pos.shape[0]
    z0 = (limits["lower"][2] if is_inverted else limits["upper"][2]) * 100
    homes = formations.cone(n, spacing, z0, delta_height * (1 if is_inverted else -1))
    ids = list(range(n))
    return _hold(_assign(swarm_pos, homes, ids, swarm_vel, tend - tstart), ids, tstart, tend)


def center(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Small ring at the swarm centroid height, held (subset-capable)."""
    ids = _sanitize_drone_ids(params[0], swarm_pos.shape[0])
    n = len(ids)
    z = float(np.mean(swarm_pos, axis=0)[2])
    homes = formations.ring(n, _ring_radius_floor(60.0, n), z)
    return _hold(_assign(swarm_pos, homes, ids, swarm_vel, tend - tstart), ids, tstart, tend)


def line_form(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Evenly spaced line of length ``length_cm`` through the swarm centroid, held."""
    (length_cm,) = params
    n = swarm_pos.shape[0]
    c = np.mean(swarm_pos, axis=0)
    start = c - np.array([length_cm / 2, 0.0, 0.0])
    end = c + np.array([length_cm / 2, 0.0, 0.0])
    ids = list(range(n))
    homes = formations.line(n, start, end)
    return _hold(_assign(swarm_pos, homes, ids, swarm_vel, tend - tstart), ids, tstart, tend)


def grid_form(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Centered rectangular grid at the swarm centroid, held."""
    (spacing,) = params
    n = swarm_pos.shape[0]
    c = np.mean(swarm_pos, axis=0)
    homes = formations.grid(n, float(spacing), c[2], c[:2])
    ids = list(range(n))
    return _hold(_assign(swarm_pos, homes, ids, swarm_vel, tend - tstart), ids, tstart, tend)


def vee_form(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """V / flying-wedge formation, apex at the swarm centroid, held."""
    spread_deg, spacing = params
    n = swarm_pos.shape[0]
    apex = np.mean(swarm_pos, axis=0)
    homes = formations.vee(n, apex, np.deg2rad(spread_deg), float(spacing))
    ids = list(range(n))
    return _hold(_assign(swarm_pos, homes, ids, swarm_vel, tend - tstart), ids, tstart, tend)


def polygon_form(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Drones distributed along a regular ``n_sides``-gon perimeter, held."""
    n_sides, radius, height = params
    n = swarm_pos.shape[0]
    homes = formations.polygon(n, int(n_sides), float(radius), float(height))
    ids = list(range(n))
    return _hold(_assign(swarm_pos, homes, ids, swarm_vel, tend - tstart), ids, tstart, tend)


def helix_form(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Static helix arrangement (points frozen along a helix), held."""
    radius, pitch, turns = params
    n = swarm_pos.shape[0]
    z0 = float(np.mean(swarm_pos, axis=0)[2])
    homes = formations.helix_static(n, float(radius), z0, float(pitch), float(turns))
    ids = list(range(n))
    return _hold(_assign(swarm_pos, homes, ids, swarm_vel, tend - tstart), ids, tstart, tend)


def _rotating_arcs(
    homes: NDArray, center: NDArray, dphi: float, t0: float, t1: float, plane: str = "xy"
) -> SplineDict:
    """Place a per-drone circular arc for a rigid rotation of a constant formation.

    Args:
        homes: Home positions in cm, shape ``(n, 3)`` (assigned).
        center: Rotation center in cm.
        dphi: Angular sweep in radians.
        t0: Block start time in seconds.
        t1: Block end time in seconds.
        plane: Rotation plane: ``"xy"`` (about z), ``"xz"`` (about y), or ``"yz"`` (about x).

    Returns:
        Drone index -> arc spline. Each drone keeps its out-of-plane coordinate.
    """
    ax = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[plane]
    out: SplineDict = {}
    for idx in range(len(homes)):
        rel = homes[idx] - center
        u, v, w = rel[ax[0]], rel[ax[1]], rel[ax[2]]
        radius = float(np.hypot(u, v))
        phi0 = float(np.arctan2(v, u))
        flat = arc_spline(np.zeros(3), radius, phi0, dphi, t0, t1)
        # Map planar arc (x=u-axis, y=v-axis) back into 3-D, restoring the fixed w-axis.
        segs = []
        for seg in flat.segments:
            cp = np.zeros((seg.degree + 1, 3))
            cp[:, ax[0]] = seg.control_points[:, 0] + center[ax[0]]
            cp[:, ax[1]] = seg.control_points[:, 1] + center[ax[1]]
            cp[:, ax[2]] = center[ax[2]] + w
            segs.append(Spline(cp, seg.t0, seg.t1))
        out[idx] = PiecewiseSpline(segs)
    return out


def _scale_in_plane(splines: SplineDict, center_xy: NDArray, scale: Spline) -> SplineDict:
    """Scale each curve's in-plane offset from ``center_xy`` by a scalar spline (z fixed).

    Args:
        splines: Drone id -> arc curve to scale.
        center_xy: In-plane scaling center ``(x, y)`` in cm.
        scale: A scalar (dim=1) spline of the radial scale factor over time.

    Returns:
        Drone id -> radially scaled curve.
    """
    out: SplineDict = {}
    for idx, curve in splines.items():
        segs = curve.segments if isinstance(curve, PiecewiseSpline) else [curve]
        scaled = []
        for seg in segs:
            seg_scale = Spline(scale.evaluate(np.array([seg.t0, seg.t1])), seg.t0, seg.t1)
            xy = Spline(seg.control_points[:, :2] - center_xy, seg.t0, seg.t1)
            scaled_xy = seg_scale * xy
            z = Spline(seg.control_points[:, 2:3], seg.t0, seg.t1).degree_elevate(scaled_xy.degree)
            cp = np.zeros((scaled_xy.degree + 1, 3))
            cp[:, :2] = scaled_xy.control_points + center_xy
            cp[:, 2] = z.control_points[:, 0]
            scaled.append(Spline(cp, seg.t0, seg.t1))
        out[idx] = PiecewiseSpline(scaled) if len(scaled) > 1 else scaled[0]
    return out


def _add_translation(splines: SplineDict, translate: Spline) -> SplineDict:
    """Add a single-segment translation spline to each (possibly piecewise) curve.

    Args:
        splines: Drone id -> curve to translate.
        translate: A degree-1 3-D translation spline over the block.

    Returns:
        Drone id -> translated curve.
    """
    out: SplineDict = {}
    for idx, curve in splines.items():
        segs = curve.segments if isinstance(curve, PiecewiseSpline) else [curve]
        moved = [
            seg + Spline(translate.evaluate(np.array([seg.t0, seg.t1])), seg.t0, seg.t1)
            for seg in segs
        ]
        out[idx] = PiecewiseSpline(moved) if len(moved) > 1 else moved[0]
    return out


def rotate(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Rotate the current layout about z by ``angle`` degrees."""
    angle, _axis = params
    c = np.mean(swarm_pos, axis=0)
    return _rotating_arcs(
        swarm_pos, np.array([0.0, 0.0, c[2]]), np.deg2rad(float(angle)), tstart, tend, "xy"
    )


def spiral(
    params: tuple,
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict,
    swarm_vel: NDArray | None = None,
    growth: float = 2.0,
    degrees: float = 360.0,
) -> SplineDict:
    """Ring rotating about z while its radius ramps outward (folds ``spiral_speed``).

    Args:
        params: ``(steps, height)`` primitive parameters.
        swarm_pos: Current swarm positions in cm, shape ``(n, 3)``.
        tstart: Block start time in seconds.
        tend: Block end time in seconds.
        limits: Workspace limits dict with ``lower``/``upper`` bounds (meters).
        swarm_vel: Optional per-drone velocities in cm/s.
        growth: Radial growth factor applied to the start radius.
        degrees: Total angular sweep in degrees.

    Returns:
        Drone id -> outward-spiraling arc trajectory.
    """
    _steps, height = params[0], params[1]
    n = swarm_pos.shape[0]
    r0 = _ring_radius_floor(60.0, n)
    r1 = min(growth * r0, limits["upper"][0] * 100)
    layout = formations.ring(n, r0, float(height))

    def build(pos: NDArray) -> SplineDict:
        arcs = _rotating_arcs(
            pos, np.array([0.0, 0.0, height]), np.deg2rad(degrees), tstart, tend, "xy"
        )
        return _scale_in_plane(arcs, np.array([0.0, 0.0]), linear_scale(1.0, r1 / r0, tstart, tend))

    return _assign_to_motion(swarm_pos, swarm_vel, layout, list(range(n)), tstart, tend, build)


def helix(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Ring rotating about z while rising in z (arc(x,y) + z-ramp)."""
    _steps, delta_h, height = params
    n = swarm_pos.shape[0]
    layout = formations.ring(n, _ring_radius_floor(60.0, n), float(height))

    def build(pos: NDArray) -> SplineDict:
        arcs = _rotating_arcs(pos, np.array([0.0, 0.0, height]), 2 * np.pi, tstart, tend, "xy")
        return _add_translation(
            arcs, linear_translate(np.zeros(3), np.array([0.0, 0.0, float(delta_h)]), tstart, tend)
        )

    return _assign_to_motion(swarm_pos, swarm_vel, layout, list(range(n)), tstart, tend, build)


def twister(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Spinning skewed-helix layout, rotating about z by ``omega``."""
    _steps, omega, z_spacing = params
    n = swarm_pos.shape[0]
    layout = formations.helix_static(
        n, min(400.0, limits["upper"][0] * 100), 100 * limits["lower"][2], float(z_spacing), 2.0
    )
    dphi = min(omega / 10.0, 2.0) * (tend - tstart)

    def build(pos: NDArray) -> SplineDict:
        return _rotating_arcs(pos, np.array([0.0, 0.0, pos[:, 2].mean()]), dphi, tstart, tend, "xy")

    return _assign_to_motion(swarm_pos, swarm_vel, layout, list(range(n)), tstart, tend, build)


def orbit(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Translate the whole formation rigidly along a circular path (orientation fixed)."""
    angle, radius = params
    c = np.mean(swarm_pos, axis=0)
    orbit_center = c + np.array([float(radius), 0.0, 0.0])
    path = _rotating_arcs(
        np.array([c]), orbit_center, np.deg2rad(float(angle)), tstart, tend, "xy"
    )[0]
    out: SplineDict = {}
    for i in range(swarm_pos.shape[0]):
        offset = swarm_pos[i] - c
        segs = [Spline(seg.control_points + offset, seg.t0, seg.t1) for seg in path.segments]
        out[i] = PiecewiseSpline(segs)
    return out


def tumble(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Rotate the current layout about the x or y axis (vertical-plane arc)."""
    angle, axis = params
    plane = "yz" if "x" in axis else "xz"
    return _rotating_arcs(
        swarm_pos, np.mean(swarm_pos, axis=0), np.deg2rad(float(angle)), tstart, tend, plane
    )


def move_z(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Translate a drone subset along z by ``distance`` cm over the block."""
    drone_ids, distance = params
    ids = _sanitize_drone_ids(drone_ids, swarm_pos.shape[0])
    return {
        d: linear_translate(
            swarm_pos[d], swarm_pos[d] + np.array([0.0, 0.0, float(distance)]), tstart, tend
        )
        for d in ids
    }


def move(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Translate one drone to an absolute position ``(x, y, z)`` cm over the block.

    ``drone_id`` is 1-indexed (the LLM-facing convention).
    """
    x, y, z, drone_id = params
    d = int(drone_id) - 1
    return {
        d: linear_translate(swarm_pos[d], np.array([float(x), float(y), float(z)]), tstart, tend)
    }


def swap(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Exchange the positions of two drones over the block (ids are 1-indexed)."""
    a, b = int(params[0]) - 1, int(params[1]) - 1
    return {
        a: linear_translate(swarm_pos[a], swarm_pos[b], tstart, tend),
        b: linear_translate(swarm_pos[b], swarm_pos[a], tstart, tend),
    }


def translate(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Rigidly drift the whole swarm by ``(dx, dy, dz)`` cm over the block."""
    delta = np.array([float(params[0]), float(params[1]), float(params[2])])
    return {
        i: linear_translate(swarm_pos[i], swarm_pos[i] + delta, tstart, tend)
        for i in range(swarm_pos.shape[0])
    }


def scale(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Expand/contract the formation about its centroid by ``factor`` (in-plane)."""
    (factor,) = params
    c = np.mean(swarm_pos, axis=0)
    matrix_end = np.diag([float(factor), float(factor), 1.0])
    out: SplineDict = {}
    for i in range(swarm_pos.shape[0]):
        disp = affine_offset(swarm_pos[i], c, matrix_end, tstart, tend)
        out[i] = constant_spline(swarm_pos[i], tstart, tend) + disp
    return out


def shear(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Ramp a shear about the centroid: axis-pair ``"xz"`` adds ``k*z`` to ``x``, etc."""
    k, axis_pair = params
    src, dst = {"xz": (2, 0), "yz": (2, 1), "xy": (1, 0)}[axis_pair]
    matrix_end = np.eye(3)
    matrix_end[dst, src] = float(k)
    c = np.mean(swarm_pos, axis=0)
    out: SplineDict = {}
    for i in range(swarm_pos.shape[0]):
        disp = affine_offset(swarm_pos[i], c, matrix_end, tstart, tend)
        out[i] = constant_spline(swarm_pos[i], tstart, tend) + disp
    return out


def zig_zag(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Grid formation translating in an alternating zig-zag while rising in z."""
    steps, delta, delta_h = params
    n = swarm_pos.shape[0]
    c = np.mean(swarm_pos, axis=0)
    layout = formations.grid(n, 50.0, c[2], c[:2])
    dxy = np.array([abs(delta), abs(delta), 0.0])
    dz = np.array([0.0, 0.0, float(delta_h)])

    def build(pos: NDArray) -> SplineDict:
        return {i: zigzag_translate(pos[i], int(steps), dxy, dz, tstart, tend) for i in range(n)}

    return _assign_to_motion(swarm_pos, swarm_vel, layout, list(range(n)), tstart, tend, build)


def _field_block(
    homes: NDArray,
    ids: list[int],
    amplitude: NDArray,
    phase: NDArray,
    periods: int,
    t0: float,
    t1: float,
) -> SplineDict:
    """Compose a sine field onto held homes (R + D, M = I).

    Args:
        homes: Held home positions in cm, shape ``(n, 3)``.
        ids: Drone ids in the same row order as ``homes``.
        amplitude: Per-drone amplitude vectors ``(n, 3)`` in cm.
        phase: Per-drone phase offsets ``(n,)`` in radians.
        periods: Number of full oscillation periods over the block.
        t0: Block start time in seconds.
        t1: Block end time in seconds.

    Returns:
        Drone id -> composed (home + sine field) trajectory.
    """
    field = fields.sine_field(homes, t0, t1, periods, amplitude, phase)
    home_map = {ids[r]: homes[r] for r in range(len(ids))}
    field_map = {ids[r]: field[r] for r in range(len(ids))}
    return compose_block(home_map, [field_map], None, t0, t1)


def wave(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Grid formation with a vertical standing wave (amplitude varies across x)."""
    _steps, height = params
    n = swarm_pos.shape[0]
    c = np.mean(swarm_pos, axis=0)
    layout = formations.grid(n, 50.0, max(float(height), 150.0), c[:2])

    def build(pos: NDArray) -> SplineDict:
        span = float(np.ptp(pos[:, 0])) or 1.0
        spatial = np.sin(np.pi * (pos[:, 0] - pos[:, 0].min()) / span)
        amp = np.zeros((n, 3))
        amp[:, 2] = 25.0 * spatial
        return _field_block(pos, list(range(n)), amp, np.zeros(n), 1, tstart, tend)

    return _assign_to_motion(swarm_pos, swarm_vel, layout, list(range(n)), tstart, tend, build)


def ripple(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Vertical wave whose phase grows with radius from the centroid (a ripple out)."""
    amp_cm, periods = params
    n = swarm_pos.shape[0]
    c = np.mean(swarm_pos, axis=0)
    r = np.linalg.norm(swarm_pos[:, :2] - c[:2], axis=1)
    phase = -2 * np.pi * r / (float(np.ptp(r)) or 1.0)
    amp = np.zeros((n, 3))
    amp[:, 2] = float(amp_cm)
    return _field_block(swarm_pos, list(range(n)), amp, phase, int(periods), tstart, tend)


def traveling_wave(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Vertical wave whose phase grows with x (propagates across the swarm)."""
    amp_cm, periods = params
    n = swarm_pos.shape[0]
    x = swarm_pos[:, 0]
    phase = -2 * np.pi * (x - x.min()) / (float(np.ptp(x)) or 1.0)
    amp = np.zeros((n, 3))
    amp[:, 2] = float(amp_cm)
    return _field_block(swarm_pos, list(range(n)), amp, phase, int(periods), tstart, tend)


def pulse(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Radial in/out oscillation: each drone moves along its radial direction."""
    amp_cm, periods = params
    n = swarm_pos.shape[0]
    c = np.mean(swarm_pos, axis=0)
    rel = swarm_pos[:, :2] - c[:2]
    direction = rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-9)
    amp = np.zeros((n, 3))
    amp[:, :2] = float(amp_cm) * direction
    return _field_block(swarm_pos, list(range(n)), amp, np.zeros(n), int(periods), tstart, tend)


def cascade(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Vertical bump phase-delayed by drone index (a sequential ripple)."""
    amp_cm, periods = params
    n = swarm_pos.shape[0]
    phase = -2 * np.pi * np.arange(n) / n
    amp = np.zeros((n, 3))
    amp[:, 2] = float(amp_cm)
    return _field_block(swarm_pos, list(range(n)), amp, phase, int(periods), tstart, tend)


def breathe(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Oscillating expand/contract about the centroid (amplitude scales with radius)."""
    max_factor, periods = params
    n = swarm_pos.shape[0]
    c = np.mean(swarm_pos, axis=0)
    amp = (float(max_factor) - 1.0) * (swarm_pos - c)
    amp[:, 2] = 0.0
    return _field_block(swarm_pos, list(range(n)), amp, np.zeros(n), int(periods), tstart, tend)


def twist(params, swarm_pos, tstart, tend, limits, swarm_vel=None):  # noqa: ANN001, ANN201
    """Rotational field: each drone rotates about the vertical axis by ``angle`` degrees.

    Conceptually a field, but rotating an offset is exactly an arc, so it composes as a
    transform (returns absolute position) rather than an additive displacement.
    """
    (angle,) = params
    return _rotating_arcs(
        swarm_pos, np.array([0.0, 0.0, 0.0]), np.deg2rad(float(angle)), tstart, tend, "xy"
    )


SPLINE_PRIMITIVE_N_ARGS: dict[str, int] = {
    # formations (held)
    "form_circle": 4,   # drone_ids, radius_cm, z_cm, time_to_finish_s
    "form_star": 4,     # height_cm, min_spacing_cm, delta_radius_cm, time_to_finish_s
    "form_cone": 4,     # delta_height_cm, spacing_cm, is_inverted, time_to_finish_s
    "center": 1,        # drone_ids
    "line": 1,          # length_cm
    "grid": 1,          # spacing_cm
    "vee": 2,           # spread_deg, spacing_cm
    "polygon": 3,       # n_sides, radius_cm, height_cm
    "helix_static": 3,  # radius_cm, pitch_cm, turns
    # motions
    "rotate": 2,        # angle_deg, axis
    "spiral": 2,        # steps, height_cm
    "spiral_speed": 4,  # steps, height_cm, degrees, radius_increase
    "helix": 3,         # steps, delta_height_cm, height_cm
    "twister": 3,       # steps, omega_times_ten, z_spacing_cm
    "orbit": 2,         # angle_deg, radius_cm
    "tumble": 2,        # angle_deg, axis
    "move_z": 2,        # drone_ids, delta_cm
    "move": 4,          # x_cm, y_cm, z_cm, drone_id
    "swap": 2,          # drone_id_1, drone_id_2
    "translate": 3,     # dx_cm, dy_cm, dz_cm
    "scale": 1,         # factor
    "shear": 2,         # k, axis_pair
    "zig_zag": 3,       # steps, delta_xy_cm, delta_z_cm
    # fields
    "wave": 2,          # steps, height_cm
    "ripple": 2,        # amp_cm, periods
    "traveling_wave": 2,  # amp_cm, periods
    "pulse": 2,         # amp_cm, periods
    "cascade": 2,       # amp_cm, periods
    "breathe": 2,       # max_factor, periods
    "twist": 1,         # angle_deg
}

SPLINE_PRIMITIVES: dict[str, Callable[..., SplineDict]] = {
    "form_circle": form_circle,
    "form_star": form_star,
    "form_cone": form_cone,
    "center": center,
    "line": line_form,
    "grid": grid_form,
    "vee": vee_form,
    "polygon": polygon_form,
    "helix_static": helix_form,
    "rotate": rotate,
    "spiral": spiral,
    "spiral_speed": spiral,
    "helix": helix,
    "twister": twister,
    "orbit": orbit,
    "tumble": tumble,
    "move_z": move_z,
    "move": move,
    "swap": swap,
    "zig_zag": zig_zag,
    "translate": translate,
    "scale": scale,
    "shear": shear,
    "wave": wave,
    "ripple": ripple,
    "traveling_wave": traveling_wave,
    "pulse": pulse,
    "cascade": cascade,
    "breathe": breathe,
    "twist": twist,
}


def spline_primitive_by_name(name: str) -> Callable[..., SplineDict]:
    """Return the WS1 spline-builder for ``name`` (``spiral_speed`` aliases ``spiral``).

    Args:
        name: The primitive name to look up.

    Returns:
        The builder callable registered under ``name``.

    Raises:
        KeyError: If ``name`` is not a supported primitive.
    """
    if name not in SPLINE_PRIMITIVES:
        raise KeyError(f"Unknown spline primitive {name}")
    return SPLINE_PRIMITIVES[name]

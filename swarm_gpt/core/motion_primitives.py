"""Motion primitive library."""

import re
import sys
from types import EllipsisType
from typing import Callable

import minsnap_trajectories as ms
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R

from swarm_gpt.exception import LLMFormatError

motion_primitives = {
    "move": {"n_args": 4},
    "rotate": {"n_args": 2},
    "center": {"n_args": 1},
    "swap": {"n_args": 2},
    "move_z": {"n_args": 2},
    "spiral": {"n_args": 2},
    "spiral_speed": {"n_args": 4},
    "helix": {"n_args": 3},
    "plan": {"n_args": 1},
    "form_circle": {"n_args": 4},
    "zig_zag": {"n_args": 3},
    "wave": {"n_args": 2},
    "twister": {"n_args": 3},
    "form_star": {"n_args": 4},
    "form_cone": {"n_args": 4},
}


def primitive_by_name(
    name: str,
) -> Callable[
    [tuple, NDArray, float, float, dict[str, NDArray]],
    tuple[NDArray, dict[float, dict[int, NDArray]]],
]:
    """Return a motion primitive by its name."""
    if name not in motion_primitives:
        raise KeyError(f"Unknown motion primitive {name}")
    return getattr(sys.modules[__name__], name)


def rotate(
    params: tuple[int, str],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Rotate all drones by angle theta."""
    angle, axis = params
    angle = np.deg2rad(float(angle))
    steps = max(1, min(int(tend - tstart), 2))
    if "z" in axis:
        axis = np.array([0, 0, 1])
    elif "y" in axis:
        axis = np.array([0, 1, 0])
    elif "x" in axis:
        axis = np.array([1, 0, 0])
    else:
        raise LLMFormatError("Invalid axis for rotation")
    max_radius = np.max(np.linalg.norm(swarm_pos[..., :2], axis=-1))
    vmax = 1.0  # m/s
    max_angle = (vmax * 100) / max_radius * (tend - tstart)
    angle = np.clip(angle, -max_angle, max_angle)
    r = R.identity() if steps == 0 else R.from_rotvec(axis * angle / steps)

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        swarm_pos = r.apply(swarm_pos)
        waypoints[t] = {i: p.copy() for i, p in enumerate(swarm_pos)}
    return swarm_pos, waypoints


def spiral(
    params: tuple[int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Spiral primitive."""
    n_drones = swarm_pos.shape[0]
    steps, height = params
    min_spacing = 60  # cm

    # Chord formula: the radius whose circumference seats every drone min_spacing apart.
    start_radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    end_radius = min(2 * start_radius, limits["upper"][0] * 100)
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    x = start_radius * np.cos(angles)
    y = start_radius * np.sin(angles)
    # TODO: Vary height over time?
    des_pos = np.array([x, y, [height] * n_drones]).T
    assignment = _assign_positions(swarm_pos, des_pos, swarm_vel=swarm_vel)
    dt = (tend - tstart) / steps

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        radius = start_radius + (end_radius - start_radius) * ((t - tstart) / (tend - tstart))
        # Whichever is slower: a full revolution, or the 100 cm/s linear velocity drone limit.
        rot_rate = min(100 / radius, 2 * np.pi / (tend - tstart))
        angles += rot_rate * dt
        swarm_pos = np.array(
            [radius * np.cos(angles), radius * np.sin(angles), [height] * n_drones]
        ).T[assignment]
        waypoints[t] = {i: p.copy() for i, p in enumerate(swarm_pos)}
    return swarm_pos, waypoints


def spiral_speed(
    params: tuple[int, int, int, float],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Spiral primitive with speed control."""
    steps, height, degrees, increase = params
    n_drones = swarm_pos.shape[0]
    min_spacing = 60  # cm
    steps = max(1, min(int(tend - tstart), 2))

    # Chord formula: the radius whose circumference seats every drone min_spacing apart.
    start_radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    end_radius = min(increase * start_radius, limits["upper"][0] * 100)
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    x = start_radius * np.cos(angles)
    y = start_radius * np.sin(angles)
    des_pos = np.array([x, y, [height] * n_drones]).T
    assignment = _assign_positions(swarm_pos, des_pos, swarm_vel=swarm_vel)
    dt = (tend - tstart) / steps

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        radius = start_radius + (end_radius - start_radius) * ((t - tstart) / (tend - tstart))
        # Whichever is slower: the requested sweep, or the 100 cm/s linear velocity drone limit.
        rot_rate = min(100 / radius, np.deg2rad(degrees) / (tend - tstart))
        angles += rot_rate * dt
        des_pos = np.array(
            [radius * np.cos(angles), radius * np.sin(angles), [height] * n_drones]
        ).T[assignment]
        waypoints[t] = {i: p.copy() for i, p in enumerate(des_pos)}

    return des_pos, waypoints


def zig_zag(
    params: tuple[int, int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Move drones in a zigzag pattern, with ``params`` of ``[steps, delta, delta_h]``.

    ``delta`` is the horizontal displacement per step and ``delta_h`` the vertical one.
    """
    steps, delta, delta_h = params
    delta = abs(delta)
    delta_xy = np.abs(np.array([delta, delta, 0]))
    delta_z = np.array([0, 0, delta_h])

    waypoints = {}
    pos = swarm_pos.copy()
    for i, t in enumerate(np.linspace(tstart, tend, steps + 1)[1:]):
        if i == 0:
            pos = _form_grid(swarm_pos, limits=limits)
            waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}
            continue
        displacement_factor = (-1) ** i
        pos += displacement_factor * delta_xy + delta_z
        waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}

    return pos, waypoints


def helix(
    params: tuple[int, int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Rise the drones up while they circle around the center."""
    steps, delta_h, height = params
    n_drones = swarm_pos.shape[0]
    min_spacing = 60  # cm
    # Chord formula: the radius whose circumference seats every drone min_spacing apart.
    radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    des_pos = np.array([x, y, [height] * n_drones]).T
    assignment = _assign_positions(swarm_pos, des_pos)
    vmax = 100  # cm/s
    rot_rate = min(vmax / radius, 2 * np.pi / (tend - tstart))
    dt = (tend - tstart) / steps

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        z = height + (t - tstart) / (tend - tstart) * delta_h
        z = min(z, limits["upper"][2] * 100)
        angles += rot_rate * dt
        pos = np.array([radius * np.cos(angles), radius * np.sin(angles), [z] * n_drones]).T[
            assignment
        ]
        waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}

    return pos, waypoints


def wave(
    params: tuple[int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Run a standing-wave pattern over a grid, with ``params`` of ``[steps, height_cm]``."""
    steps, height = params
    steps = int(steps)
    # TODO: Tune default values
    a = 100.0  # Rectangle length
    b = 100.0  # Rectangle width
    c = np.pi  # Speed of wave propagation
    a_mu = np.array([[0.0, 0.0, 0.25]])  # (N, 3)
    b_mu = np.array([[0.0, 0.0, 0.25]])  # (N, 3)
    mu1_mu2 = np.array([[0.4, 0.4]])  # (N, 2)
    height = max(height, 150)  # Floored to stay out of ground effect.

    # Frequencies dictated by dispersion relation
    omega = c * np.pi * np.sqrt((mu1_mu2[:, 0] ** 2) / a**2 + (mu1_mu2[:, 1] ** 2) / b**2)

    grid_time = np.linspace(tstart, tend, steps + 1)[1]
    waypoints = {}
    swarm_pos = _form_grid(swarm_pos, limits=limits, height=height, spacing=50)
    waypoints[grid_time] = {i: p.copy() for i, p in enumerate(swarm_pos)}

    start_pos = swarm_pos.copy()
    for t in np.linspace(tstart, tend, steps + 1)[2:]:
        sin_mu1 = np.sin(mu1_mu2[None, :, 0] / a * np.pi * start_pos[:, [0]])  # (n_drones, N)
        sin_mu2 = np.sin(mu1_mu2[None, :, 1] / b * np.pi * start_pos[:, [1]])  # (n_drones, N)
        sin2_term = sin_mu1 * sin_mu2  # (n_drones, N)
        sin_omega_t = np.sin(omega * t)  # (N, )
        cos_omega_t = np.cos(omega * t)  # (N, )
        u_terms = sin2_term[..., None] * (
            a_mu[None, ...] * sin_omega_t + b_mu[None, ...] * cos_omega_t
        )
        # (n_drones, N, 3)
        u = u_terms.sum(axis=1) * 100  # TODO: Remove the 100 factor for scaling to cm
        swarm_pos = start_pos + u
        waypoints[t] = {i: p.copy() for i, p in enumerate(swarm_pos)}

    return swarm_pos, waypoints


def form_star(
    params: tuple[int, int, int, float],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a star shape with the drones with {n_drones}//2 spokes."""
    height, min_spacing, delta_radius, time_to_finish_s = params
    min_spacing = max(min_spacing, 40)
    delta_radius = max(delta_radius, 40)
    n_drones = swarm_pos.shape[0]
    drones_per_circle = n_drones // 2
    height = int(height)

    # Chord formula: the radius whose circumference seats every drone min_spacing apart.
    radius = min_spacing / (2 * np.sin(np.pi / drones_per_circle))

    radii = [radius, radius + delta_radius]
    angle_offset = [0, 2 * np.pi / drones_per_circle]

    des_pos = None
    for r, offset in zip(radii, angle_offset):
        angles = np.linspace(0, 2 * np.pi, drones_per_circle, endpoint=False) + offset
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        if des_pos is None:
            des_pos = np.array([x, y, [height] * drones_per_circle]).T
        else:
            des_pos = np.vstack([des_pos, np.array([x, y, [height] * drones_per_circle]).T])
    # An odd swarm leaves one drone over; it goes in the centre.
    if n_drones != drones_per_circle * 2:
        des_pos = np.vstack([des_pos, np.array([0, 0, height]).T])

    assignment = _assign_positions(swarm_pos, des_pos, swarm_vel=swarm_vel)
    target = des_pos[assignment]
    waypoints = _formation_waypoints(target, swarm_pos, tstart, tend, time_to_finish_s)
    return target, waypoints


def form_cone(
    params: tuple[int, int, bool, float],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a cone with the drones."""
    delta_height, spacing, is_inverted, time_to_finish_s = params
    n_drones = swarm_pos.shape[0]

    start_height = (limits["lower"][2] if is_inverted else limits["upper"][2]) * 100
    delta_height = delta_height * (1 if is_inverted else -1)

    drones_left = n_drones
    drone_increase_per_layer = 4

    radius = 0
    z = start_height
    des_pos = np.array([0, 0, z]).T
    drones_left -= 1

    drones_in_layer = 0
    while drones_left > 0:
        drones_in_layer += drone_increase_per_layer
        z += delta_height
        radius = spacing / (2 * np.sin(np.pi / drones_in_layer))

        drones_left -= drones_in_layer
        if drones_left < 0:
            drones_in_layer = drones_left + drones_in_layer

        angles = np.linspace(0, 2 * np.pi, drones_in_layer, endpoint=False)

        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        des_pos = np.vstack([des_pos, np.array([x, y, [z] * drones_in_layer]).T])

    assignment = _assign_positions(swarm_pos, des_pos, swarm_vel=swarm_vel)
    target = des_pos[assignment]
    waypoints = _formation_waypoints(target, swarm_pos, tstart, tend, time_to_finish_s)
    return target, waypoints


def twister(
    params: tuple[int, int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a spinning upside-down cone with drones."""
    steps, omega, z_spacing = params
    n_drones = swarm_pos.shape[0]
    # LLM will output omega that is 10x to avoid decimals. TODO: Change this
    omega = omega / 10
    max_omega = 2
    omega = min(omega, max_omega)

    lim_lower, lim_upper = limits["lower"], limits["upper"]
    min_spacing = 60  # cm
    turns = 2  # Full revolutions the helix winds through
    # The drones are spread evenly along the winding, so consecutive ones sit
    # 2*pi*turns/(n_drones - 1) apart in angle and the innermost pair is the tightest. Size the
    # inner radius from that step the way the ring primitives do, rather than from a fixed 30cm
    # that only cleared the collision envelope for a handful of drones. Capped at a half turn so
    # a swarm small enough to put neighbours on opposite sides does not inflate the cone.
    half_step = min(np.pi * turns / max(n_drones - 1, 1), np.pi / 2)
    min_radius = min_spacing / (2 * np.sin(half_step))
    # The cone opens out to fill the arena. Small swarms lean on that taper for their spacing:
    # at n_drones - 1 <= turns the angular step is a whole revolution, so every drone lands on
    # one ray and only the radial step keeps them apart. Floored at the inner radius so a swarm
    # too large for the arena keeps its spacing and reaches past the limits, as helix does.
    arena_radius = np.min(lim_upper[:2] - lim_lower[:2]) * 100 / 2
    max_radius = max(float(arena_radius), min_radius)

    z_center = 100 * (lim_lower[2] + (lim_upper[2] - lim_lower[2]) / 2)
    max_height = min(z_center + z_spacing * n_drones / 2, lim_upper[2] * 100)
    min_height = max(z_center - z_spacing * n_drones / 2, lim_lower[2] * 100)

    radius = np.linspace(min_radius, max_radius, n_drones)
    z = np.linspace(min_height, max_height, n_drones)
    angles = np.linspace(0, 2 * np.pi * turns, n_drones)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    des_pos = np.array([x, y, z]).T

    assignment = _assign_positions(swarm_pos, des_pos)
    dt = (tend - tstart) / steps

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        angles += omega * dt
        pos = np.array([radius * np.cos(angles), radius * np.sin(angles), z]).T[assignment]
        waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}

    return pos, waypoints


def center(
    params: tuple[str | list[int]],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Move all the drones to the center, calculated from current position."""
    drone_ids = _sanitize_drone_ids(params[0], swarm_pos.shape[0])
    n_drones = len(drone_ids)
    centroid = np.mean(swarm_pos, axis=0)
    min_spacing = 60  # cm
    # Chord formula: the radius whose circumference seats every drone min_spacing apart.
    radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    des_pos = np.array([x, y, [centroid[2]] * n_drones]).T
    vel_subset = swarm_vel[drone_ids] if swarm_vel is not None else None
    assignment = _assign_positions(swarm_pos[drone_ids], des_pos, swarm_vel=vel_subset)
    waypoints = {}
    waypoints[tend] = {i: p.copy() for i, p in enumerate(des_pos[assignment])}
    pos = swarm_pos.copy()
    pos[drone_ids] = des_pos[assignment]
    return pos, waypoints


def form_circle(
    params: tuple[str | list[int], int, int, float],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Position drones around the circumference of a circle at a given radius and height."""
    drone_ids, radius_cm, z_coord_cm, time_to_finish_s = params
    drone_ids = _sanitize_drone_ids(drone_ids, swarm_pos.shape[0])
    n_drones = len(drone_ids)
    z_coord = int(z_coord_cm)
    min_spacing = 80  # cm
    min_radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    # Respect LLM-specified radius but enforce minimum safe spacing
    radius = max(float(radius_cm), min_radius)
    lim_upper, lim_lower = limits["upper"], limits["lower"]
    max_diameter = min(lim_upper[0] - lim_lower[0], lim_upper[1] - lim_lower[1])
    max_radius = max_diameter * 100 / 2

    radii = [radius]
    drones_per_circle = [n_drones]
    if radius > max_radius:
        n_drones_outer = int(np.pi / np.asin(min_spacing / (2 * max_radius)))
        n_drones_inner = n_drones - n_drones_outer
        radius_outer = max_radius
        radius_inner = min_spacing / (2 * np.sin(np.pi / n_drones_inner))
        radii = [radius_outer, radius_inner]
        drones_per_circle = [n_drones_outer, n_drones_inner]

    des_pos = None
    for r, n in zip(radii, drones_per_circle):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        if des_pos is None:
            des_pos = np.array([x, y, [z_coord] * n]).T
        else:
            des_pos = np.vstack([des_pos, np.array([x, y, [z_coord] * n]).T])

    vel_subset = swarm_vel[drone_ids] if swarm_vel is not None else None
    assignment = _assign_positions(swarm_pos[drone_ids], des_pos, swarm_vel=vel_subset)
    target = des_pos[assignment]
    waypoints = _formation_waypoints(
        target, swarm_pos[drone_ids], tstart, tend, time_to_finish_s, drone_ids=drone_ids
    )
    pos = swarm_pos.copy()
    pos[drone_ids] = target
    return pos, waypoints


def swap(
    params: tuple[int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Swap the positions of two drones."""
    drone1_id, drone2_id = params
    drone1_id, drone2_id = drone1_id - 1, drone2_id - 1
    waypoints = {}
    pos = swarm_pos.copy()
    waypoints[tend] = {drone1_id: pos[drone2_id].copy(), drone2_id: pos[drone1_id].copy()}
    pos[drone1_id], pos[drone2_id] = pos[drone2_id].copy(), pos[drone1_id].copy()
    return pos, waypoints


def move_z(
    params: tuple[str | list[int], int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Move the drones along the z-axis."""
    drone_ids, distance = params
    drone_ids = _sanitize_drone_ids(drone_ids, swarm_pos.shape[0])
    steps = max(1, min(int(tend - tstart), 2))

    z_min = limits["lower"][2] * 100
    z_max = limits["upper"][2] * 100
    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        swarm_pos[drone_ids, 2] = np.clip(swarm_pos[drone_ids, 2] + distance / steps, z_min, z_max)
        waypoints[t] = {i: swarm_pos[i].copy() for i in drone_ids}

    return swarm_pos, waypoints


def move(
    params: tuple[float, float, float, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
    swarm_vel: NDArray | None = None,
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Translate move function to waypoints."""
    x, y, z, drone_id = params
    drone_id = drone_id - 1
    swarm_pos[drone_id] = np.array([x, y, z])
    return swarm_pos, {tend: {drone_id: np.array([x, y, z])}}


def _form_grid(
    swarm_pos: NDArray,
    limits: dict[str, NDArray],
    height: float | None = None,
    spacing: int | None = None,
) -> NDArray:
    """Form a grid of drones at the current position."""
    n_drones = swarm_pos.shape[0]
    rows = int(np.sqrt(n_drones))
    cols = int(np.ceil(n_drones / rows))
    min_spacing = 50
    spacing = min_spacing if spacing is None else max(spacing, min_spacing)
    x, y = np.meshgrid(np.arange(cols) * spacing, np.arange(rows) * spacing)
    lim_upper, lim_lower = limits["upper"], limits["lower"]
    assert (x.max() - x.min()) / 100 <= lim_upper[0] - lim_lower[0], "Grid too wide"
    assert (y.max() - y.min()) / 100 <= lim_upper[1] - lim_lower[1], "Grid too tall"
    x = (x.flatten() - x.mean())[:n_drones]
    y = (y.flatten() - y.mean())[:n_drones]
    centroid = np.mean(swarm_pos, axis=0)
    z = np.full(n_drones, max(10, min(200, centroid[2] if height is None else height)))
    x, y = x + centroid[0], y + centroid[1]
    if (dx := x.max() - lim_upper[0] * 100) > 0:
        x -= dx
    if (dy := y.max() - lim_upper[1] * 100) > 0:
        y -= dy
    if (dx := x.min() - lim_lower[0] * 100) < 0:
        x -= dx
    if (dy := y.min() - lim_lower[1] * 100) < 0:
        y -= dy
    des_pos = np.stack([x, y, z], axis=1)
    assignment = _assign_positions(swarm_pos, des_pos)
    return des_pos[assignment]


# Comma-separated 1-indexed ids and inclusive ranges, e.g. "1-50" or "1-20,31,45-60". Also the
# `pattern` the structured-output schema puts on `drone_ids`, so the syntax the model is
# constrained to and the syntax the backend accepts are one string. `expand_drone_id_spec`
# additionally tolerates whitespace around the tokens, so a preset is not rejected over a space.
DRONE_ID_SPEC_PATTERN = r"^\d+(-\d+)?(,\d+(-\d+)?)*$"
_DRONE_ID_TOKEN_RE = re.compile(r"^(\d+)(?:-(\d+))?$")


def expand_drone_id_spec(spec: str) -> list[int]:
    """Expand a compact drone selection into the explicit 1-indexed ids it names, in order.

    BOTH ENDPOINTS OF A RANGE ARE INCLUSIVE: ``"1-50"`` contains drone 50, so the next block starts
    at 51. A drone named twice is rejected rather than flown to two targets at once.
    """
    if not isinstance(spec, str):
        raise LLMFormatError(f"Drone IDs must be a range string like '1-50', got {spec}")
    ids: list[int] = []
    seen: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        match = _DRONE_ID_TOKEN_RE.match(token)
        if match is None:
            raise LLMFormatError(
                f"Drone ID selection '{spec}' is malformed at '{token}'. Write comma-separated "
                "ids and inclusive ranges, e.g. '7', '1-50' or '1-20,31,45-60'"
            )
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if start > end:
            raise LLMFormatError(
                f"Drone ID range '{token}' in '{spec}' runs backwards. Write it as '{end}-{start}'"
            )
        if start < 1:
            raise LLMFormatError(f"Drone IDs are 1-indexed, but '{spec}' names drone 0")
        block = range(start, end + 1)
        if repeated := sorted(seen.intersection(block)):
            raise LLMFormatError(
                f"Drone ID selection '{spec}' names drone(s) {repeated} more than once. Range "
                "endpoints are inclusive, so consecutive blocks must not share one: split the "
                "swarm as '1-50' then '51-100', never '1-50' then '50-100'"
            )
        seen.update(block)
        ids.extend(block)
    return ids


def _sanitize_drone_ids(drone_ids: str | list[int], n_drones: int) -> list[int]:
    """Resolve a 1-indexed drone selection into 0-indexed swarm indices, in selection order.

    Accepts the compact spec the LLM emits (``"1-50"``), a plain 1-indexed list, or a list holding
    ``...`` for the whole swarm. Bounds on the list form are left to callers (`lighting.select`).
    """
    if isinstance(drone_ids, str):
        ids = expand_drone_id_spec(drone_ids)
        if out_of_range := sorted({i for i in ids if i > n_drones}):
            raise LLMFormatError(
                f"Drone IDs {out_of_range} in '{drone_ids}' are outside the 1..{n_drones} swarm"
            )
        return [i - 1 for i in ids]
    if not isinstance(drone_ids, list):
        raise LLMFormatError(
            f"Drone IDs must be a range string like '1-50' or a list of integers, got {drone_ids}"
        )
    if any(isinstance(i, EllipsisType) for i in drone_ids):
        return list(range(n_drones))
    if not all(isinstance(id, int) for id in drone_ids):
        raise LLMFormatError(f"Drone IDs must be a list of integers, got {drone_ids}")
    return [id - 1 for id in drone_ids]  # TODO: Make LLM assign IDs starting at 0


# Effective speed cap used when scheduling formation arrivals. Held below axswarm's
# vel_max=1.73 m/s to leave the MPC headroom; matches the convention used by `rotate`
# and `spiral`. HEADROOM is a multiplicative buffer for the solver's smoothness and
# input-continuity penalties; T_MIN prevents trivially small moves from scheduling
# zero-duration arrivals that the MPC can't track cleanly.
_FORMATION_V_EFF_MPS = 1.0
_FORMATION_HEADROOM = 1.3
_FORMATION_T_MIN_S = 0.5
# MPC lookahead window length (K=50 timesteps at freq=10 Hz). Hold waypoints are only emitted when
# the remaining interval exceeds this, so the axswarm lookahead is never empty without flooding the
# waypoints array for normal-length intervals.
_MPC_HORIZON_S = 5.0


def _minsnap_cost_matrix(pos: NDArray, des_pos: NDArray, vel: NDArray) -> NDArray:
    """Compute per-pair minimum-snap trajectory cost matrix.

    For each (drone i, target j) pair, generates a two-waypoint minimum-snap
    trajectory from drone i's current position and velocity to target j with
    zero final velocity. The time budget T_ij is derived from the per-pair
    Euclidean distance and the drone velocity cap. The snap cost
    (integral of squared snap over [0, T_ij]) is returned as the cost.

    Args:
        pos: Current drone positions in cm, shape (n, 3).
        des_pos: Desired target positions in cm, shape (m, 3).
        vel: Current drone velocities in cm/s, shape (n, 3).

    Returns:
        Cost matrix of shape (n, m).
    """
    n, m = len(pos), len(des_pos)
    cost = np.zeros((n, m))
    # Convert from cm / cm·s⁻¹ to m / m·s⁻¹ for the package
    pos_m = pos / 100.0
    des_m = des_pos / 100.0
    vel_ms = vel / 100.0

    for i in range(n):
        for j in range(m):
            dist_m = float(np.linalg.norm(des_m[j] - pos_m[i]))
            T = max(dist_m / _FORMATION_V_EFF_MPS * _FORMATION_HEADROOM, _FORMATION_T_MIN_S)
            waypoints = [
                ms.Waypoint(time=0.0, position=pos_m[i], velocity=vel_ms[i]),
                ms.Waypoint(time=T, position=des_m[j]),
            ]
            traj = ms.generate_trajectory(waypoints, degree=8, idx_minimized_orders=4)
            t_samples = np.linspace(0.0, T, 20)
            derivs = ms.compute_trajectory_derivatives(traj, t_samples, num_orders=5)
            snap = derivs[4]  # shape (20, 3)
            cost[i, j] = float(np.trapezoid(np.sum(snap**2, axis=-1), t_samples))

    return cost


def _assign_positions(pos: NDArray, des_pos: NDArray, swarm_vel: NDArray | None = None) -> NDArray:
    """Assign drones to the closest desired positions.

    Uses minimum-snap trajectory cost when ``swarm_vel`` is provided, falling
    back to Euclidean distance otherwise.

    Args:
        pos: Current drone positions in cm, shape (n, 3).
        des_pos: Desired target positions in cm, shape (m, 3).
        swarm_vel: Current drone velocities in cm/s, shape (n, 3). When
            provided, per-pair minimum-snap cost is used as the assignment
            metric instead of Euclidean distance.

    Returns:
        The assigned target indices as a numpy array of shape (n,).
    """
    if swarm_vel is not None:
        cost = _minsnap_cost_matrix(pos, des_pos, swarm_vel)
    else:
        cost = np.linalg.norm(pos[:, None, :] - des_pos[None, :, :], axis=-1)
    return linear_sum_assignment(cost)[1]


def _formation_arrival_time(
    target_pos: NDArray, current_pos: NDArray, tstart: float, tend: float
) -> float:
    """Estimate the earliest feasible arrival time, in ``[tstart, tend]``.

    Sizes the interval to the bottleneck drone's displacement at an effective max velocity, with
    headroom for MPC smoothness. Positions are in cm.
    """
    max_travel_m = float(np.linalg.norm(target_pos - current_pos, axis=-1).max()) / 100
    travel_time = max(max_travel_m / _FORMATION_V_EFF_MPS * _FORMATION_HEADROOM, _FORMATION_T_MIN_S)
    return min(tstart + travel_time, tend)


def _formation_waypoints(
    target_pos: NDArray,
    current_pos: NDArray,
    tstart: float,
    tend: float,
    time_to_finish_s: float,
    drone_ids: list[int] | None = None,
) -> dict[float, dict[int, NDArray]]:
    """Schedule a formation arrival at the LLM-chosen time, clamped to physics and the interval.

    The physics floor is ``_formation_arrival_time``'s duration. Holds are emitted only when the
    remaining interval exceeds ``_MPC_HORIZON_S``, i.e. when axswarm's lookahead would be empty.
    """
    ids = drone_ids if drone_ids is not None else list(range(len(target_pos)))
    physics_min_duration = _formation_arrival_time(target_pos, current_pos, tstart, tend) - tstart
    duration = float(np.clip(time_to_finish_s, physics_min_duration, tend - tstart))
    arrival = tstart + duration
    entry = {d: p.copy() for d, p in zip(ids, target_pos)}
    waypoints: dict[float, dict[int, NDArray]] = {arrival: entry}
    t = arrival + _MPC_HORIZON_S
    while t < tend:
        waypoints[t] = {d: p.copy() for d, p in zip(ids, target_pos)}
        t += _MPC_HORIZON_S
    if arrival < tend:
        waypoints[tend] = {d: p.copy() for d, p in zip(ids, target_pos)}
    return waypoints

"""Motion primitive library."""

import sys
from types import EllipsisType
from typing import Callable

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
    "form_circle": {"n_args": 3},
    "zig_zag": {"n_args": 3},
    "wave": {"n_args": 5},
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
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Rotate all drones by angle theta."""
    angle, axis = params
    angle = np.deg2rad(float(angle))
    steps = int(tend - tstart)  # Number of steps to rotate
    # override rotation to be around z axis atm
    if "z" in axis:
        axis = np.array([0, 0, 1])
    elif "y" in axis:
        axis = np.array([0, 1, 0])
    elif "x" in axis:
        axis = np.array([1, 0, 0])
    else:
        raise LLMFormatError("Invalid axis for rotation")
    max_radius = np.max(np.linalg.norm(swarm_pos[..., :2], axis=-1))
    vmax = 1.0  # Maximum velocity in m/s
    max_angle = (vmax * 100) / max_radius * (tend - tstart)
    angle = np.clip(angle, -max_angle, max_angle)
    r = R.identity() if steps == 0 else R.from_rotvec(axis * angle / steps)

    # Apply the rotation to the vector
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
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Spiral primitive."""
    n_drones = swarm_pos.shape[0]
    steps, height = params
    # steps = 4
    min_spacing = 60  # Minimum distance between drones in cm

    # Calculate the circumference needed to place all drones with at least the minimum spacing
    start_radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    end_radius = min(2 * start_radius, limits["upper"][0] * 100)
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    # Match start positions to drones
    x = start_radius * np.cos(angles)
    y = start_radius * np.sin(angles)
    # TODO: Vary height over time?
    des_pos = np.array([x, y, [height] * n_drones]).T
    assignment = _assign_positions(swarm_pos, des_pos)
    dt = (tend - tstart) / steps

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        radius = start_radius + (end_radius - start_radius) * ((t - tstart) / (tend - tstart))
        # Either full rotation around the circle or max angular velocity with 100cm/s linear
        # velocity hard-coded as drone limit
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
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Spiral primitive with speed control."""
    steps, height, degrees, increase = params
    n_drones = swarm_pos.shape[0]
    min_spacing = 60  # Minimum distance between drones in cm
    steps = int(tend - tstart)

    # Calculate the circumference needed to place all drones with at least the minimum spacing
    start_radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    end_radius = min(increase * start_radius, limits["upper"][0] * 100)
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    # Match start positions to drones
    x = start_radius * np.cos(angles)
    y = start_radius * np.sin(angles)
    des_pos = np.array([x, y, [height] * n_drones]).T
    assignment = _assign_positions(swarm_pos, des_pos)
    dt = (tend - tstart) / steps

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        radius = start_radius + (end_radius - start_radius) * ((t - tstart) / (tend - tstart))
        # Either full rotation around the circle or max angular velocity with 100cm/s linear
        # velocity hard-coded as drone limit
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
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Moves drones in a zigzag pattern.

    Params:
        params: [steps, delta, delta_h]
        steps: Number of steps (an integer).
        delta: Horizontal displacement per step (an integer).
        delta_h: Vertical displacement per step (an integer).
    """
    steps, delta, delta_h = params
    delta = abs(delta)  # Ensure delta is positive for displacement
    delta_xy = np.abs(np.array([delta, delta, 0]))
    delta_z = np.array([0, 0, delta_h])

    waypoints = {}
    pos = swarm_pos.copy()
    for i, t in enumerate(np.linspace(tstart, tend, steps + 1)[1:]):
        if i == 0:
            pos = _form_grid(swarm_pos, limits=limits)
            waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}
            continue
        displacement_factor = (-1) ** i  # Alternates between 1 and -1
        pos += displacement_factor * delta_xy + delta_z
        waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}

    return pos, waypoints


def helix(
    params: tuple[int, int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Helix primitive.

    Drones rise up and circle around the center at the same time.
    """
    steps, delta_h, height = params
    n_drones = swarm_pos.shape[0]
    min_spacing = 60  # Minimum distance between drones in cm
    # Calculate the circumference needed to place all drones with at least the minimum spacing
    radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    # Match start positions to drones
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    des_pos = np.array([x, y, [height] * n_drones]).T
    assignment = _assign_positions(swarm_pos, des_pos)
    vmax = 100  # Maximum velocity in cm/s
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
    params: tuple[int, int, list[tuple[float, float]], list[float], list[float]],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Specific wave pattern.

    Args:
        params: [steps, height, µ_pairs, aµ1µ2, bµ1µ2]
        swarm_pos: Current positions of the drones.
        tstart: Start time of the primitive.
        tend: End time of the primitive.
        limits: Spatial limits for the drones.
    """
    steps, height, mu_pairs, a_mu, b_mu = params
    steps = int(steps)
    # TODO: Tune default values
    a = 100.0  # Rectangle length
    b = 100.0  # Rectangle Width
    c = np.pi  # Speed of wave propagation
    a_mu = np.array([[0.0, 0.0, 0.25]])  # Shape: (N, 3)
    b_mu = np.array([[0.0, 0.0, 0.25]])  # Shape: (N, 3)
    mu1_mu2 = np.array([[0.4, 0.4]])  # Shape: (N, 2)
    height = max(height, 150)  # Restrict to 75cm for ground effect avoidance

    # Frequencies dictated by dispersion relation
    omega = c * np.pi * np.sqrt((mu1_mu2[:, 0] ** 2) / a**2 + (mu1_mu2[:, 1] ** 2) / b**2)

    # Arrange all drones in a grid like formation
    grid_time = np.linspace(tstart, tend, steps + 1)[1]
    # First step is to form a grid
    waypoints = {}
    swarm_pos = _form_grid(swarm_pos, limits=limits, height=height, spacing=50)
    waypoints[grid_time] = {i: p.copy() for i, p in enumerate(swarm_pos)}

    start_pos = swarm_pos.copy()
    for t in np.linspace(tstart, tend, steps + 1)[2:]:
        # Calculate all sum terms vectorized
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
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a star shape with the drones with {n_drones}//2 spokes."""
    height, min_spacing, delta_radius, time_to_finish_s = params
    min_spacing = max(min_spacing, 40)
    delta_radius = max(delta_radius, 40)
    n_drones = swarm_pos.shape[0]
    drones_per_circle = n_drones // 2
    height = int(height)

    # Calculate the circumference needed to place all drones with at least the minimum spacing
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
    # If odd number of drones, put the drone at the center
    if n_drones != drones_per_circle * 2:
        des_pos = np.vstack([des_pos, np.array([0, 0, height]).T])

    assignment = _assign_positions(swarm_pos, des_pos)
    target = des_pos[assignment]
    waypoints = _formation_waypoints(target, swarm_pos, tstart, tend, time_to_finish_s)
    return target, waypoints


def form_cone(
    params: tuple[int, int, bool, float],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a cone with the drones."""
    delta_height, spacing, is_inverted, time_to_finish_s = params
    n_drones = swarm_pos.shape[0]

    # Define limits
    start_height = (limits["lower"][2] if is_inverted else limits["upper"][2]) * 100
    delta_height = delta_height * (1 if is_inverted else -1)

    drones_left = n_drones
    drone_increase_per_layer = 4

    # Place first drone
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

    assignment = _assign_positions(swarm_pos, des_pos)
    target = des_pos[assignment]
    waypoints = _formation_waypoints(target, swarm_pos, tstart, tend, time_to_finish_s)
    return target, waypoints


def twister(
    params: tuple[int, int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a spinning upside-down cone with drones."""
    steps, omega, z_spacing = params
    n_drones = swarm_pos.shape[0]
    # LLM will output omega that is 10x to avoid decimals. TODO: Change this
    omega = omega / 10
    max_omega = 2
    omega = min(omega, max_omega)  # Restrict angular velocity

    lim_lower, lim_upper = limits["lower"], limits["upper"]
    max_radius = min(np.min(lim_upper[:2] - lim_lower[:2] * 100) / 2, 400)
    min_radius = 30

    z_center = 100 * (lim_lower[2] + (lim_upper[2] - lim_lower[2]) / 2)
    max_height = min(z_center + z_spacing * n_drones / 2, lim_upper[2] * 100)
    min_height = max(z_center - z_spacing * n_drones / 2, lim_lower[2] * 100)

    # Calculate the radius and height for each drone
    radius = np.linspace(min_radius, max_radius, n_drones)
    z = np.linspace(min_height, max_height, n_drones)
    angles = np.linspace(0, 4 * np.pi, n_drones)
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
    params: tuple[list[int]],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Move all the drones to the center, calculated from current position."""
    drone_ids = _sanitize_drone_ids(params[0], swarm_pos.shape[0])
    n_drones = len(drone_ids)
    centroid = np.mean(swarm_pos, axis=0)
    min_spacing = 60  # Minimum distance between drones in cm
    # Calculate the circumference needed to place all drones with at least the minimum spacing
    radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    des_pos = np.array([x, y, [centroid[2]] * n_drones]).T
    assignment = _assign_positions(swarm_pos[drone_ids], des_pos)
    waypoints = {}
    waypoints[tend] = {i: p.copy() for i, p in enumerate(des_pos[assignment])}
    pos = swarm_pos.copy()
    pos[drone_ids] = des_pos[assignment]
    return pos, waypoints


def form_circle(
    params: tuple[list[int], int, float],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Position drones around the circumference of a circle at height z with at least {min_spacing} cm apart."""
    drone_ids, z_coord, time_to_finish_s = params
    drone_ids = _sanitize_drone_ids(drone_ids, swarm_pos.shape[0])
    n_drones = len(drone_ids)
    z_coord = int(z_coord)  # z coordinate in cm
    min_spacing = 80  # Minimum distance between drones in cm
    # Calculate the circumference needed to place all drones with at least the minimum spacing
    # If radius is bigger than the limits, make concentric circles
    radius = min_spacing / (2 * np.sin(np.pi / n_drones))
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

    assignment = _assign_positions(swarm_pos[drone_ids], des_pos)
    target = des_pos[assignment]
    waypoints = _formation_waypoints(target, swarm_pos[drone_ids], tstart, tend, time_to_finish_s)
    pos = swarm_pos.copy()
    pos[drone_ids] = target
    return pos, waypoints


def swap(
    params: tuple[int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
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
    params: tuple[list[int], int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Move the drones along the z-axis."""
    drone_ids, distance = params
    drone_ids = _sanitize_drone_ids(drone_ids, swarm_pos.shape[0])
    steps = int(tend - tstart)

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        swarm_pos[drone_ids, 2] = np.clip(swarm_pos[drone_ids, 2] + distance / steps, 100, 200)
        waypoints[t] = {i: swarm_pos[i].copy() for i in drone_ids}

    return swarm_pos, waypoints


def move(
    params: tuple[float, float, float, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
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
    # Get the number of rows and columns
    n_drones = swarm_pos.shape[0]
    rows = int(np.sqrt(n_drones))
    cols = int(np.ceil(n_drones / rows))
    # Get the spacing between the drones
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


def _sanitize_drone_ids(drone_ids: list[int], n_drones: int) -> list[int]:
    if not isinstance(drone_ids, list):
        raise LLMFormatError(f"Drone IDs must be a list of integers, got {drone_ids}")
    if any(isinstance(i, EllipsisType) for i in drone_ids):
        return list(range(n_drones))
    if not all(isinstance(id, int) for id in drone_ids):
        raise LLMFormatError(f"Drone IDs must be a list of integers, got {drone_ids}")
    return [id - 1 for id in drone_ids]  # TODO: Make LLM assign IDs starting at 0


def _assign_positions(pos: NDArray, des_pos: NDArray) -> NDArray:
    """Assign drones to the closest desired positions.

    Returns:
        The assigned IDs as a numpy array.
    """
    # Get the distance matrix
    dist = np.linalg.norm(pos[:, None, :] - des_pos[None, :, :], axis=-1)
    # Use the Hungarian algorithm to find the optimal assignment
    return linear_sum_assignment(dist)[1]


# Effective speed cap used when scheduling formation arrivals. Held below axswarm's
# vel_max=1.73 m/s to leave the MPC headroom; matches the convention used by `rotate`
# and `spiral`. HEADROOM is a multiplicative buffer for the solver's smoothness and
# input-continuity penalties; T_MIN prevents trivially small moves from scheduling
# zero-duration arrivals that the MPC can't track cleanly.
_FORMATION_V_EFF_MPS = 1.0
_FORMATION_HEADROOM = 1.3
_FORMATION_T_MIN_S = 0.5
# Spacing between hold waypoints once a formation is reached. Must stay below
# axswarm's K/freq=5s lookahead (settings.yaml: K=50, freq=10). With a gap >= the
# horizon, _filter_horizon returns an empty mask and axswarm clips past waypoints
# into the current step, yanking drones back to stale positions.
_FORMATION_HOLD_DT_S = 4.0


def _formation_arrival_time(
    target_pos: NDArray, current_pos: NDArray, tstart: float, tend: float
) -> float:
    """Estimate the earliest feasible arrival time for a formation primitive.

    Sizes the interval to the bottleneck drone's displacement at an effective max
    velocity (with headroom for MPC smoothness), then clamps to ``[tstart, tend]``.

    Args:
        target_pos: Desired post-assignment drone positions in cm, shape (n, 3).
        current_pos: Current positions of the same drones in cm, shape (n, 3).
        tstart: Interval start time in seconds.
        tend: Interval end time in seconds.

    Returns:
        Arrival time in seconds, in ``[tstart, tend]``.
    """
    max_travel_m = float(np.linalg.norm(target_pos - current_pos, axis=-1).max()) / 100
    travel_time = max(max_travel_m / _FORMATION_V_EFF_MPS * _FORMATION_HEADROOM, _FORMATION_T_MIN_S)
    return min(tstart + travel_time, tend)


def _formation_waypoints(
    target_pos: NDArray, current_pos: NDArray, tstart: float, tend: float, time_to_finish_s: float
) -> dict[float, dict[int, NDArray]]:
    """Schedule a formation arrival at the LLM-chosen time (clamped to physics + interval).

    The physics floor is the existing ``_formation_arrival_time`` duration; ``time_to_finish_s``
    is clamped between that floor and the full interval. Hold waypoints are then emitted every
    ``_FORMATION_HOLD_DT_S`` seconds so axswarm's 5-second lookahead is never empty.

    Args:
        target_pos: Desired post-assignment drone positions in cm, shape (n, 3).
        current_pos: Current positions of the same drones in cm, shape (n, 3).
        tstart: Interval start time in seconds.
        tend: Interval end time in seconds.
        time_to_finish_s: LLM-requested arrival duration in seconds. Clamped to
            ``[physics_min_duration, tend - tstart]``.

    Returns:
        ``{time: {drone_id: pos}}`` waypoints covering ``[arrival, tend]``.
    """
    physics_min_duration = _formation_arrival_time(target_pos, current_pos, tstart, tend) - tstart
    duration = float(np.clip(time_to_finish_s, physics_min_duration, tend - tstart))
    arrival = tstart + duration
    waypoints: dict[float, dict[int, NDArray]] = {
        arrival: {i: p.copy() for i, p in enumerate(target_pos)}
    }
    t = arrival + _FORMATION_HOLD_DT_S
    while t < tend:
        waypoints[t] = {i: p.copy() for i, p in enumerate(target_pos)}
        t += _FORMATION_HOLD_DT_S
    if arrival < tend:
        waypoints[tend] = {i: p.copy() for i, p in enumerate(target_pos)}
    return waypoints

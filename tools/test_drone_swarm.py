"""Manual hardware smoke test for DroneSwarm.

This script requires real Crazyflie hardware. In mocap mode it also requires ROS2
and the configured TF names to be available.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import make_interp_spline

from swarm_gpt.core.drone_swarm import DroneSwarm

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drones-file",
        type=Path,
        default=REPO_ROOT / "swarm_gpt" / "data" / "drones.toml",
        help="Path to the TOML drone config.",
    )
    parser.add_argument(
        "--lighthouse",
        action="store_true",
        help="Use lighthouse state estimates instead of mocap observations.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N drones.")
    parser.add_argument("--height", type=float, default=0.6, help="Takeoff/test height in meters.")
    parser.add_argument("--min-height", type=float, default=0.2, help="Minimum accepted z after takeoff.")
    parser.add_argument("--takeoff-duration", type=float, default=3.0)
    parser.add_argument("--goto-duration", type=float, default=3.0)
    parser.add_argument("--choreo-duration", type=float, default=3.0)
    parser.add_argument("--land-duration", type=float, default=3.0)
    parser.add_argument("--goto-dx", type=float, default=0.15, help="Small x offset for goto test.")
    parser.add_argument("--goto-dy", type=float, default=0.0, help="Small y offset for goto test.")
    parser.add_argument("--ctrl-freq", type=float, default=50.0)
    parser.add_argument("--update-freq", type=float, default=10.0)
    parser.add_argument("--col-freq", type=float, default=10.0)
    parser.add_argument(
        "--emergency-stop",
        action="store_true",
        help="Also call emergency_stop() after landing. This requires rebooting the drones.",
    )
    parser.add_argument("--yes", action="store_true", help="Do not prompt before flight.")
    return parser.parse_args()


def load_drones(path: Path, limit: int | None) -> dict[str, dict[str, Any]]:
    """Load the drone table from TOML."""
    with path.open("rb") as file:
        drones = tomllib.load(file)
    if limit is None:
        return drones
    return dict(list(drones.items())[:limit])


def init_ros_if_needed(lighthouse: bool) -> None:
    """Initialize ROS2 for mocap mode."""
    if lighthouse:
        return
    import rclpy

    if not rclpy.ok():
        rclpy.init()


def check_observations(swarm: DroneSwarm) -> None:
    """Check that every active drone has finite localization data."""
    print("\nObservation checks")
    for uri in swarm.uris:
        if not swarm.is_active(uri):
            print(f"  {uri}: inactive")
            continue

        obs = swarm.get_obs(uri)
        pos = np.asarray(obs["pos"], dtype=float)
        quat = np.asarray(obs["quat"], dtype=float)
        rpy = np.asarray(obs["rpy"], dtype=float)
        if pos.shape != (3,) or quat.shape != (4,) or rpy.shape != (3,):
            raise RuntimeError(f"Invalid observation shape for {uri}: {obs}")
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)) or not np.all(np.isfinite(rpy)):
            raise RuntimeError(f"Non-finite observation for {uri}: {obs}")
        if np.linalg.norm(quat) < 0.5:
            raise RuntimeError(f"Suspicious quaternion for {uri}: {quat}")
        print(
            f"  {uri}: pos = {pos.round(3).tolist()}, "
            f"rpy = {np.degrees(rpy).round(1).tolist()}"
        )


def pose_for_uri(
    swarm: DroneSwarm,
    drones_by_uri: dict[str, dict[str, Any]],
    uri: str,
    height: float,
) -> np.ndarray:
    """Return a [x, y, z, yaw_deg] pose for command references."""
    if not swarm.is_active(uri):
        base = np.asarray(drones_by_uri[uri]["pos"], dtype=float)
        return np.array([base[0], base[1], height, 0.0])

    obs = swarm.get_obs(uri)
    pos = np.asarray(obs["pos"], dtype=float)
    yaw_deg = np.degrees(float(obs["rpy"][2]))
    return np.array([pos[0], pos[1], max(pos[2], height), yaw_deg])


def build_color_refs(swarm: DroneSwarm) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build one simple color command per configured drone."""
    color_top = {}
    color_bot = {}
    for i, uri in enumerate(swarm.uris):
        hue = np.array(
            [
                0.0,
                80.0 if i % 3 == 0 else 0.0,
                80.0 if i % 3 == 1 else 0.0,
                50.0 if i % 3 == 2 else 0.0,
            ]
        )
        color_top[uri] = hue
        color_bot[uri] = hue
    return color_top, color_bot


def assert_airborne(swarm: DroneSwarm, min_height: float) -> None:
    """Check that active drones are above the minimum height."""
    low = []
    for uri in swarm.uris:
        if not swarm.is_active(uri):
            continue
        z = swarm.get_obs(uri)["pos"][2]
        if z < min_height:
            low.append((uri, float(z)))
    if low:
        raise RuntimeError(f"Drones below minimum height after takeoff: {low}")


def run_smoke_test(args: argparse.Namespace) -> None:
    """Run observation and flight checks on real hardware."""
    drones = load_drones(args.drones_file, args.limit)
    if not drones:
        raise RuntimeError(f"No drones found in {args.drones_file}")

    init_ros_if_needed(args.lighthouse)
    drones_by_uri = {drone["uri"]: drone for drone in drones.values()}

    print(f"Loaded {len(drones)} drones from {args.drones_file}")
    print(f"Localization: {'lighthouse' if args.lighthouse else 'mocap'}")

    swarm = DroneSwarm(
        drones,
        ctrl_freq=args.ctrl_freq,
        update_freq=args.update_freq,
        col_freq=args.col_freq,
        lighthouse=args.lighthouse,
    )

    try:
        print(f"Active drones: {sorted(swarm.active_uris)}")
        print(f"Missing drones: {swarm.missing_uris()}")
        check_observations(swarm)

        print("\nResetting and applying colors")
        swarm.reset()
        color_top, color_bot = build_color_refs(swarm)
        swarm.apply_colors(color_top, color_bot)

        if not args.yes:
            input("\nPress Enter to start takeoff/goto/choreography/land, or Ctrl+C to abort...")

        print("\nTaking off")
        swarm.takeoff(height=args.height, duration=args.takeoff_duration)
        try:
            assert_airborne(swarm, args.min_height)
        except RuntimeError:
            print("\nTakeoff check failed; landing active drones before aborting")
            swarm.land(height=0.0, duration=args.land_duration)
            raise
        takeoff_poses = {
            uri: pose_for_uri(swarm, drones_by_uri, uri, args.height) for uri in swarm.uris
        }

        print("\nGoing to a small offset")
        goto_refs = {}
        goto_targets = {}
        for uri in swarm.uris:
            target = takeoff_poses[uri].copy()
            target[0] += args.goto_dx
            target[1] += args.goto_dy
            goto_targets[uri] = target
            goto_refs[uri] = [target]
        swarm.goto(goto_refs, duration=args.goto_duration)

        print("\nExecuting simple choreography")
        choreography = {}
        cue_color_top = {}
        cue_color_bot = {}
        for uri, target in goto_targets.items():
            start = target[:3]
            end = takeoff_poses[uri][:3]
            choreography[uri] = make_interp_spline(
                [0.0, args.choreo_duration], np.vstack([start, end]), k=1, axis=0
            )
            cue_color_top[uri] = {
                0.0: np.array([0.0, 60.0, 0.0, 0.0]),
                args.choreo_duration / 2.0: np.array([0.0, 0.0, 60.0, 0.0]),
            }
            cue_color_bot[uri] = {
                0.0: np.array([0.0, 0.0, 0.0, 60.0]),
                args.choreo_duration / 2.0: np.array([0.0, 60.0, 60.0, 0.0]),
            }
        swarm.execute_choreography(
            choreography,
            args.choreo_duration,
            color_top=cue_color_top,
            color_bot=cue_color_bot,
        )

        print("\nLanding")
        swarm.land(height=0.0, duration=args.land_duration)

        if args.emergency_stop:
            print("\nSending emergency stop")
            swarm.emergency_stop()

    finally:
        print("\nClosing swarm")
        swarm.close()


def main() -> None:
    """Script entry point."""
    run_smoke_test(parse_args())


if __name__ == "__main__":
    main()

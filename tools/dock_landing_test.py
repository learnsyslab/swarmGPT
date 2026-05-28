"""Minimal Crazyflie dock landing test.

Run from the deploy environment. This script intentionally bypasses the
SwarmGPT frontend, LLM choreography, and simulation layers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import toml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "swarm_gpt/data/drones.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test landing one Crazyflie on a dock.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--drone", default="cf0", help="Key in drones.toml, for example cf0.")
    parser.add_argument("--uri", help="Override the URI from drones.toml.")
    parser.add_argument(
        "--dock-pos",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Dock touchdown position in the mocap/world frame. Defaults to drone pos in config.",
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="Landing yaw reference in degrees.")
    parser.add_argument("--hover-height", type=float, default=0.6)
    parser.add_argument("--takeoff-duration", type=float, default=3.0)
    parser.add_argument("--land-duration", type=float, default=4.0)
    parser.add_argument(
        "--method",
        choices=["bitcraze-land", "goto-land"],
        default="bitcraze-land",
        help="Use Bitcraze high-level land() or repo low-level goto() descent.",
    )
    parser.add_argument(
        "--lighthouse",
        action="store_true",
        help="Use onboard Lighthouse estimates instead of mocap external pose updates.",
    )
    parser.add_argument(
        "--connect-only",
        action="store_true",
        help="Only open and close a cflib link. Does not arm or fly.",
    )
    parser.add_argument("--yes", action="store_true", help="Actually arm and fly.")
    return parser.parse_args()


def load_drone(config: Path, drone_key: str, uri_override: str | None) -> dict[str, dict]:
    drones = toml.load(config)
    if drone_key not in drones:
        raise KeyError(f"{drone_key!r} not found in {config}")

    drone = dict(drones[drone_key])
    if uri_override is not None:
        drone["uri"] = uri_override
    return {drone_key: drone}


def check_link(uri: str) -> None:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

    cflib.crtp.init_drivers()
    print(f"Opening link to {uri}...")
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")):
        print(f"Connected to {uri}")
    print("Link closed.")


def run_landing_test(args: argparse.Namespace) -> None:
    import rclpy

    from swarm_gpt.core.drone_swarm import DroneSwarm

    if not rclpy.ok():
        rclpy.init()

    drones = load_drone(args.config, args.drone, args.uri)
    uri = next(iter(drones.values()))["uri"]
    dock_pos = np.array(args.dock_pos or next(iter(drones.values()))["pos"], dtype=float)
    yaw = np.deg2rad(args.yaw)

    print(f"Connecting to {uri} with {'Lighthouse' if args.lighthouse else 'mocap'} positioning...")
    swarm = DroneSwarm(drones, lighthouse=args.lighthouse)
    try:
        print(f"Taking off to {args.hover_height:.2f} m...")
        swarm.takeoff(height=args.hover_height, duration=args.takeoff_duration)
        time.sleep(1.0)

        above_dock = np.array([dock_pos[0], dock_pos[1], args.hover_height, yaw])
        print(f"Moving above dock at {above_dock[:3]}...")
        swarm.goto({uri: [above_dock]}, duration=3.0)
        time.sleep(1.0)

        if args.method == "bitcraze-land":
            print("Landing with Bitcraze high-level commander land()...")
            swarm.land(height=dock_pos[2], duration=args.land_duration)
        else:
            touchdown = np.array([dock_pos[0], dock_pos[1], dock_pos[2], yaw])
            print(f"Landing with low-level goto() descent to {touchdown[:3]}...")
            swarm.goto({uri: [touchdown]}, duration=args.land_duration)

        time.sleep(0.5)
    finally:
        print("Closing link and stopping setpoints...")
        swarm.close()


def main() -> None:
    args = parse_args()
    drones = load_drone(args.config, args.drone, args.uri)
    uri = next(iter(drones.values()))["uri"]

    if args.connect_only:
        check_link(uri)
        return

    if not args.yes:
        raise SystemExit("Refusing to fly without --yes. Use --connect-only for a no-flight check.")

    run_landing_test(args)


if __name__ == "__main__":
    main()

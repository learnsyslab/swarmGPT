"""Bench calibration for the lighting palette, gamma and deck orientation.

Drives `DroneSwarm.apply_colors` directly: no choreographer, no axswarm, no takeoff. Run it with
the drones sitting on the docks and the radios up. See `--help` for the four modes.
"""

from __future__ import annotations

import argparse
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from numpy.typing import NDArray

from swarm_gpt.core.lighting import hue_to_wrgb, load_lighting_config

if TYPE_CHECKING:
    from collections.abc import Iterator

    from swarm_gpt.core.drone_swarm import DroneSwarm
    from swarm_gpt.core.lighting import LightingConfig

# One bench step: a label for the operator, plus the {uri: wrgb} each deck is set to.
Frame = tuple[str, dict[str, NDArray], dict[str, NDArray]]

# Deck-check colours, picked so a swap is unmistakable rather than a subtle hue shift.
_DECK_TOP = "red"
_DECK_BOT = "blue"


def _load_drones(only: str | None) -> tuple[dict[str, dict], dict]:
    """Read the active drone table and settings straight from the data files.

    Deliberately not via `Choreographer`, which would drag the LLM and music-analysis stack into a
    bench tool whose only need is a radio URI per drone.
    """
    data = Path(__file__).resolve().parents[1] / "swarm_gpt/data"
    with open(data / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(data / "drones.toml", "rb") as f:
        raw = tomllib.load(f)

    uri_base = settings["radio"]["uri_base"]
    registry = {name: entry for name, entry in raw.items() if name != "active"}
    drones = {
        name: {
            "addr": registry[name]["addr"],
            "uri": uri_base.format(channel=registry[name]["channel"], addr=registry[name]["addr"]),
            "pos": registry[name]["pos"],
        }
        for name in raw["active"]
    }
    if only is None:
        return drones, settings
    wanted = [name.strip() for name in only.split(",") if name.strip()]
    if missing := [name for name in wanted if name not in drones]:
        raise SystemExit(f"Not in the active drone list: {missing}. Active: {sorted(drones)}")
    return {name: drones[name] for name in wanted}, settings


def _apply_brightness(colour: NDArray, brightness: float, cfg: LightingConfig) -> NDArray:
    """Apply the read-out's b_min floor, bucket quantization and gamma to one colour.

    Mirrors `LightingTimeline.evaluate`, so what the bench shows is what the engine would send.
    """
    merged = 0.0 if brightness < cfg.b_min else brightness
    quantized = np.floor(merged * cfg.brightness_steps) / cfg.brightness_steps
    return np.round(colour * quantized**cfg.gamma)


def _both_decks(uris: list[str], colour: NDArray) -> tuple[dict, dict]:
    """Build the ``{uri: wrgb}`` pair that puts one colour on both decks of every drone."""
    return ({uri: colour for uri in uris}, {uri: colour for uri in uris})


def mode_deck(uris: list[str], cfg: LightingConfig, args: argparse.Namespace) -> Iterator[Frame]:
    """Light the two decks differently so their physical orientation can be read off."""
    top, bot = cfg.palette[_DECK_TOP], cfg.palette[_DECK_BOT]
    label = f"top={_DECK_TOP}  bot={_DECK_BOT}  (if the rings look swapped, the mapping is wrong)"
    yield label, {uri: top for uri in uris}, {uri: bot for uri in uris}


def mode_palette(uris: list[str], cfg: LightingConfig, args: argparse.Namespace) -> Iterator[Frame]:
    """Walk every named palette entry across both decks, one at a time."""
    for name, colour in cfg.palette.items():
        wrgb = " ".join(f"{c:5.1f}" for c in colour)
        yield f"{name:<8} WRGB [{wrgb}]", *_both_decks(uris, colour)


def mode_wheel(uris: list[str], cfg: LightingConfig, args: argparse.Namespace) -> Iterator[Frame]:
    """Spread the generated hue wheel across the swarm, which is what `channel_gain` corrects.

    Hues run in drone-id order rather than `_base_colour`'s neighbour walk, since a bench row is
    not a formation. The per-hue brightness this exposes is the same either way.
    """
    n = len(uris)
    colours = {uri: hue_to_wrgb(np.array(i / n), cfg) for i, uri in enumerate(uris)}
    yield f"hue wheel over {n} drones (look for a hue reading brighter or dimmer)", colours, colours


def mode_ramp(uris: list[str], cfg: LightingConfig, args: argparse.Namespace) -> Iterator[Frame]:
    """Step one colour down through every brightness bucket, exercising gamma, b_min and steps."""
    if args.color not in cfg.palette:
        raise SystemExit(f"Unknown colour {args.color!r}. Palette: {sorted(cfg.palette)}")
    base = cfg.palette[args.color]
    for k in range(cfg.brightness_steps, -1, -1):
        brightness = k / cfg.brightness_steps
        colour = _apply_brightness(base, brightness, cfg)
        label = f"{args.color} b={brightness:.3f} -> WRGB {colour.astype(int).tolist()}"
        yield label, *_both_decks(uris, colour)


MODES = {"deck": mode_deck, "palette": mode_palette, "wheel": mode_wheel, "ramp": mode_ramp}

# Modes whose whole point is a steady look the operator studies, so they wait rather than advance.
_WAIT_FOR_ENTER = ("deck", "wheel")


def _show(frames: Iterator[Frame], swarm: DroneSwarm | None, hold: float) -> None:
    """Apply each frame in turn, holding for ``hold`` seconds or until Enter when it is zero."""
    for label, top, bot in frames:
        print(f"  {label}")
        if swarm is None:
            continue
        swarm.apply_colors(top, bot)
        if hold > 0:
            time.sleep(hold)
        else:
            input("    [Enter] to continue ")


def main(argv: list[str] | None = None) -> None:
    """Run one bench calibration mode against the active drones."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=sorted(MODES), help="which calibration to run")
    parser.add_argument("--drones", help="comma-separated subset, e.g. cf11,cf12 (default: active)")
    parser.add_argument("--hold", type=float, help="seconds per step; 0 waits for Enter")
    parser.add_argument("--color", default="blue", help="palette colour for `ramp` (default: blue)")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the WRGB values without touching a radio"
    )
    args = parser.parse_args(argv)

    cfg = load_lighting_config()
    drones, settings = _load_drones(args.drones)
    uris = [d["uri"] for d in drones.values()]
    hold = args.hold if args.hold is not None else (0.0 if args.mode in _WAIT_FOR_ENTER else 2.0)
    print(f"{args.mode}: {len(uris)} drone(s) — {', '.join(sorted(drones))}")

    frames = MODES[args.mode](uris, cfg, args)
    if args.dry_run:
        print("  (dry run — no radio)")
        _show(frames, None, hold)
        return

    from swarm_gpt.core.drone_swarm import DroneSwarm

    # Mocap builds a ROSConnector, which asserts on an uninitialized rclpy (backend.py:302).
    if not settings["lighthouse"]:
        try:
            import rclpy
        except ImportError:
            raise SystemExit(
                "ROS2 is not available. Run this under `pixi run -e deploy`."
            ) from None
        if not rclpy.ok():
            rclpy.init()

    swarm = DroneSwarm(drones, col_freq=cfg.col_freq, lighthouse=settings["lighthouse"])
    try:
        _show(frames, swarm, hold)
    finally:
        # Leave the swarm dark rather than holding the last frame on the bench.
        swarm.apply_colors(None, None)
        swarm.close()


if __name__ == "__main__":
    main()

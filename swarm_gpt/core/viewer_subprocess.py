"""Spawned-process entry for MuJoCo live viewer.

macOS requires GLFW (and nested AppKit paths) on the real process main thread. Gradio invokes
simulate callbacks from worker threads, which triggers ``SIGTRAP`` if we open ``Sim.render`` inline.
Launching ``simulate_spline`` in a ``spawn``ed child restores the Linux/desktop behaviour.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path


def run_spline_viewer_from_payload_path(payload_path: str) -> None:
    """Unpickle payload, ``chdir`` to repo root, and run spline playback with viewer.

    This function must remain a top-level symbol so ``multiprocessing`` ``spawn`` can pickle it.

    Args:
        payload_path: Filesystem path to a pickle written by ``AppBackend``.
    """
    if sys.platform == "darwin":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    path = Path(payload_path)
    with path.open("rb") as fh:
        payload = pickle.load(fh)

    root = Path(payload["root_path"]).resolve()
    os.chdir(root)

    # Imports after chdir so ``swarm_gpt/core/sim.py`` resolves ``scene.xml``.
    from swarm_gpt.core.sim import simulate_spline
    from swarm_gpt.utils.music_manager import MusicManager

    music_dir = Path(payload["music_dir"]).resolve()
    mm = MusicManager(music_dir)
    mm.song = payload["song"]

    simulate_spline(
        payload["splines"],
        payload["settings"],
        payload["t_end"],
        mm,
        bool(payload["gui"]),
    )

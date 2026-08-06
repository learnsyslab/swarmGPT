from pathlib import Path

import numpy as np


def with_ulp_noise(points: np.ndarray, col: int = 0) -> np.ndarray:
    """Return ``points`` with its largest ``col`` coordinate nudged up two ULPs.

    The degeneracy fixtures are rings built from ``cos``/``sin``, so their radii are equal in exact
    arithmetic but equal only to within float noise in practice -- the case the lighting engine's
    relative span tolerance exists for, and one an exactly-equal fixture cannot reach, since that
    takes the degenerate branch either way and pins nothing.

    How much noise ``cos``/``sin`` actually leave is platform-dependent: linux-64 can return a
    bit-for-bit exact ring where osx-arm64 leaves a couple of ULPs. Pinning it here rather than
    assuming it keeps those fixtures degenerate-but-not-exact everywhere. Two ULPs is far below the
    engine's tolerance, so the collapse being asserted still happens.

    The nudge lands on the row where ``col`` is largest, not on a fixed row: two ULPs of a
    coordinate that happens to sit at zero is a denormal, which moves neither that column's span
    nor the point's radius.
    """
    out = np.asarray(points, dtype=float).copy()
    row = int(np.argmax(np.abs(out[:, col])))
    out[row, col] = np.nextafter(np.nextafter(out[row, col], np.inf), np.inf)
    return out


def virtual_crazyswarm_config(n_drones: int) -> Path:
    """Create a virtual crazyswarm config file for testing."""
    n_cols = int(np.ceil(np.sqrt(n_drones)))
    n_rows = int(np.ceil(n_drones / n_cols))
    spacing = 1.0

    x = np.linspace(0, (n_cols - 1) * spacing, n_cols)
    y = np.linspace(0, (n_rows - 1) * spacing, n_rows)
    X, Y = np.meshgrid(x, y)

    positions = np.zeros((n_drones, 3))
    positions[:, 0] = X.flatten()[:n_drones]
    positions[:, 1] = Y.flatten()[:n_drones]
    positions[:, 0] -= np.mean(positions[:, 0])
    positions[:, 1] -= np.mean(positions[:, 1])
    positions = np.round(positions).astype(int)

    # Assign fake cf-names cf00..cfNN and addresses 0x00..0xNN.
    # Channel = (addr // 10) * 10 = 0 for all (fine for unit tests).
    names = [f"cf{i:02d}" for i in range(n_drones)]
    active_line = "active = [" + ", ".join(f'"{n}"' for n in names) + "]\n\n"
    drone_entries = ""
    for i, (name, pos) in enumerate(zip(names, positions)):
        channel = (i // 10) * 10
        drone_entries += (
            f"[{name}]\naddr = {i}\nchannel = {channel}\npos = {pos.astype(float).tolist()}\n\n"
        )

    tmp_dir = Path("/tmp/swarm_gpt_test")
    tmp_dir.mkdir(exist_ok=True)
    config_path = tmp_dir / "drones.toml"
    config_path.write_text(active_line + drone_entries)
    return config_path

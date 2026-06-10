from pathlib import Path

import numpy as np


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
        drone_entries += f"[{name}]\naddr = {i}\npos = {pos.astype(float).tolist()}\n\n"

    tmp_dir = Path("/tmp/swarm_gpt_test")
    tmp_dir.mkdir(exist_ok=True)
    config_path = tmp_dir / "drones.toml"
    config_path.write_text(active_line + drone_entries)
    return config_path

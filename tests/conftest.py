from pathlib import Path

import numpy as np
import toml


def virtual_crazyswarm_config(n_drones: int) -> Path:
    """Create a virtual crazyswarm config file for testing."""
    # Create grid positions like in llm_success_rates.py
    n_cols = int(np.ceil(np.sqrt(n_drones)))
    n_rows = int(np.ceil(n_drones / n_cols))
    spacing = 1.0

    # Create meshgrid
    x = np.linspace(0, (n_cols - 1) * spacing, n_cols)
    y = np.linspace(0, (n_rows - 1) * spacing, n_rows)
    X, Y = np.meshgrid(x, y)

    # Flatten and combine into positions array
    positions = np.zeros((n_drones, 3))
    positions[:, 0] = X.flatten()[:n_drones]
    positions[:, 1] = Y.flatten()[:n_drones]

    # Center the grid around origin
    positions[:, 0] -= np.mean(positions[:, 0])
    positions[:, 1] -= np.mean(positions[:, 1])

    # Round to integers
    positions = np.round(positions).astype(int)

    config = {}
    for i, pos in enumerate(positions):
        config[f"cf{i}"] = {
            "name": f"cf{i}",
            "uri": f"radio://0/80/2M/E7E7E7E7{i:02X}",
            "pos": pos.astype(float).tolist(),
        }

    # Create temporary config file
    tmp_dir = Path("/tmp/swarm_gpt_test")
    tmp_dir.mkdir(exist_ok=True)
    config_path = tmp_dir / "drones.toml"

    with open(config_path, "w") as f:
        toml.dump(config, f)

    return config_path

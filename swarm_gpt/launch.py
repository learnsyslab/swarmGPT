"""Launch script for the SwarmGPT demo."""

import logging
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from datetime import datetime
from pathlib import Path

import fire

from swarm_gpt.core import AppBackend
from swarm_gpt.ui import create_ui


def mklog_date(path: Path) -> Path:
    """Make a unique directory within the given directory with the current time as name.

    Args:
        path: Parent folder path.
    """
    assert path.is_dir()
    save_file = path / (str(datetime.now().strftime("%Y_%m_%d_%H_%M")) + "_log.json")
    if not save_file.is_file():
        return save_file
    t = 1
    while save_file.is_file():
        curr_date_unique = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_({t})"
        save_file = path / (str(curr_date_unique) + "_log.json")
        t += 1
    return save_file


def main(
    strict: bool = True, model_id: str = "gpt-4o-2024-05-13", use_motion_primitives: bool = True
):
    """Build the gui and launch the demo."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress httpx info messages
    logging.getLogger("jax").setLevel(logging.WARNING)
    # logging.getLogger("swarm_gpt").setLevel(logging.DEBUG)

    # Check if the OpenAI API key is present
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY environment variable required, has not been set")
    # Get a list of all music titles available in the music directory
    music_dir = Path(__file__).resolve().parents[1] / "music"

    # Model IDs: "gpt-4o-2024-05-13", "gpt-3.5-turbo-0125", "gpt-4o-2024-05-13"
    backend = AppBackend(
        music_dir=music_dir,
        strict_processing=strict,
        model_id=model_id,
        use_motion_primitives=use_motion_primitives,
    )
    ui = create_ui(backend)
    ui.launch()


if __name__ == "__main__":
    fire.Fire(main)

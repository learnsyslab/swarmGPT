"""Launch script for the SwarmGPT demo."""

import logging
import os
import sys
from pathlib import Path
from types import FrameType

import fire
import uvicorn

from swarm_gpt.api.server import ApiConfig, create_app
from swarm_gpt.utils.llm_providers import LLMProvider, shutdown_ollama_generation

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Stable JAX on Apple Silicon (avoids some Metal/backend edge cases during MJX stepping).
if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


# models: gpt-5.4-nano, gpt-4o, gpt-4o-2024-05-13, o3-mini
def main(
    strict: bool = True,
    model_id: str = "gpt-5.4-nano",
    llm_provider: LLMProvider = "openai",
    use_motion_primitives: bool = True,
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """Launch the SwarmGPT browser app API."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress httpx info messages
    logging.getLogger("jax").setLevel(logging.WARNING)
    # logging.getLogger("swarm_gpt").setLevel(logging.DEBUG)

    if llm_provider not in ("openai", "ollama"):
        raise ValueError(f"llm_provider must be 'openai' or 'ollama', got {llm_provider!r}")
    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        logging.warning(
            "OPENAI_API_KEY is unset. OpenAI-backed runs will fail until you export it "
            "or switch the UI to Ollama (local)."
        )

    music_dir = Path(__file__).resolve().parents[1] / "music" / "songs"

    app = create_app(
        ApiConfig(
            music_dir=music_dir,
            strict_processing=strict,
            model_id=model_id,
            llm_provider=llm_provider,
            use_motion_primitives=use_motion_primitives,
        )
    )
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))
    # uvicorn captures SIGINT/SIGTERM itself (Server.capture_signals installs self.handle_exit)
    # and only starts a graceful shutdown, which waits on the open events WebSocket while the
    # deploy thread keeps flying. Wrap handle_exit so the first Ctrl+C cuts the motors before
    # uvicorn begins draining. capture_signals installs whatever self.handle_exit is at call time.
    uvicorn_handle_exit = server.handle_exit

    def handle_exit(sig: int, frame: FrameType | None) -> None:
        logging.warning("Ctrl+C received: emergency-stopping all active swarms.")
        try:
            app.state.store.emergency_stop_all()
        except Exception:
            logging.exception("Emergency stop during shutdown failed")
        uvicorn_handle_exit(sig, frame)

    server.handle_exit = handle_exit
    try:
        server.run()
    finally:
        shutdown_ollama_generation()


if __name__ == "__main__":
    fire.Fire(main)

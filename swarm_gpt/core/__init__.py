"""Backend code for interfacing with the LLMs, AMSwarm, pybullet-drones, and the crazyflies."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swarm_gpt.core.backend import AppBackend
    from swarm_gpt.core.choreographer import Choreographer

__all__ = ["AppBackend", "Choreographer"]


def __getattr__(name: str) -> Any:
    """Resolve the package's two entry points on first access (PEP 562).

    Importing them eagerly made any ``swarm_gpt.core.<module>`` import pull in the choreographer
    and reach ``structured_output_schema`` via ``music_analyzer`` -- a cycle.
    """
    if name == "AppBackend":
        from swarm_gpt.core.backend import AppBackend

        return AppBackend
    if name == "Choreographer":
        from swarm_gpt.core.choreographer import Choreographer

        return Choreographer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

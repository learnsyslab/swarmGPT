"""Core package for the swarm_gpt package.

This submodule contains the backend code for interfacing with the LLMs, AMSwarm, pybullet-drones,
and the crazyflies.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swarm_gpt.core.backend import AppBackend
    from swarm_gpt.core.choreographer import Choreographer

__all__ = ["AppBackend", "Choreographer"]


def __getattr__(name: str) -> Any:
    """Resolve the package's two entry points on first access (PEP 562).

    Importing them eagerly made *any* ``swarm_gpt.core.<module>`` import pull in the choreographer,
    which imports ``swarm_gpt.utils.music_analyzer``, which imports
    ``swarm_gpt.core.structured_output_schema`` -- a cycle that broke ``tools/analyze_songs.py``
    with a partially-initialized-module ImportError. Nothing in that chain needs the choreographer;
    only this convenience re-export did, so it is deferred until something asks for it.

    Args:
        name: Attribute being looked up on the package.

    Returns:
        The requested class.

    Raises:
        AttributeError: If the package has no such attribute.
    """
    if name == "AppBackend":
        from swarm_gpt.core.backend import AppBackend

        return AppBackend
    if name == "Choreographer":
        from swarm_gpt.core.choreographer import Choreographer

        return Choreographer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

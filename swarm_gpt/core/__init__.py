"""Core package for the swarm_gpt package.

This submodule contains the backend code for interfacing with the LLMs, AMSwarm, pybullet-drones,
and the crazyflies.
"""

from swarm_gpt.core.choreographer import Choreographer  # noqa: I001 (avoid circular import)
from swarm_gpt.core.backend import AppBackend

__all__ = ["AppBackend", "Choreographer"]

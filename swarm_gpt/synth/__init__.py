"""Runtime synthesis of motion primitives, with the safety filter as the teacher.

A research prototype for the verified-primitive-synthesis question: an LLM authors a primitive's
Python source against the same calling contract the hand-written library uses, the primitive runs,
axswarm filters it, and the resulting measurements are handed back in one of three encodings.
"""

from swarm_gpt.synth.manifest import ParamSpec, PrimitiveManifest
from swarm_gpt.synth.sandbox import SynthError, compile_invariants, compile_primitive

__all__ = [
    "ParamSpec",
    "PrimitiveManifest",
    "SynthError",
    "compile_invariants",
    "compile_primitive",
]

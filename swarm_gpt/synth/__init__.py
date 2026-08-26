"""Runtime synthesis of motion primitives, with the safety filter as the teacher.

A research prototype for the verified-primitive-synthesis question: an LLM authors the equation of
a shape, the library's own formation helpers fly the swarm into it, axswarm filters the result, and
the resulting measurements are handed back in one of three encodings.
"""

from swarm_gpt.synth.manifest import ParamSpec, PrimitiveManifest
from swarm_gpt.synth.sandbox import SynthError, compile_shape

__all__ = ["ParamSpec", "PrimitiveManifest", "SynthError", "compile_shape"]

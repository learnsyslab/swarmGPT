"""The single declaration a synthesized primitive is built from.

A primitive's signature otherwise lives in four places (prompt, structured-output schema, backend
function, offline check) and drifts between them. Deriving arity and argument order from one
manifest is what makes runtime authoring tractable at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from swarm_gpt.core.motion_primitives import register_synthesized
from swarm_gpt.core.structured_output_schema import register_synthesized_action
from swarm_gpt.synth.sandbox import SynthError, compile_shape
from swarm_gpt.synth.shape import as_primitive

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

_PARAM_TYPES = frozenset({"int", "float"})


@dataclass(frozen=True)
class ParamSpec:
    """One positional parameter of a synthesized primitive."""

    name: str
    type: str
    minimum: float
    maximum: float

    def coerce(self, value: Any) -> int | float:
        """Cast and range-check one argument value.

        Raises:
            SynthError: If the value is not numeric or falls outside ``[minimum, maximum]``.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SynthError(f"Parameter {self.name!r} must be a number, got {value!r}")
        if not self.minimum <= value <= self.maximum:
            raise SynthError(
                f"Parameter {self.name!r} = {value} is outside its declared range "
                f"[{self.minimum}, {self.maximum}]"
            )
        return int(value) if self.type == "int" else float(value)


@dataclass(frozen=True)
class PrimitiveManifest:
    """A runtime-authored primitive: its intent, signature, and the geometry it stands on.

    ``source`` is a shape -- ``(params, n_drones)`` returning one position per drone -- which
    ``synth/shape.py`` wraps into the contract the rest of the library speaks.
    """

    name: str
    intent: str
    params: tuple[ParamSpec, ...]
    source: str

    @property
    def n_args(self) -> int:
        """Number of positional arguments the primitive takes."""
        return len(self.params)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PrimitiveManifest:
        """Build a manifest from an LLM structured-output payload.

        Raises:
            SynthError: If a field is missing, mistyped, or names an unknown parameter type.
        """
        fields = ("name", "intent", "params", "source")
        missing = [k for k in fields if k not in payload]
        if missing:
            raise SynthError(f"Manifest is missing required fields: {sorted(missing)}")
        # A tuple as well as a list: an LLM payload arrives as JSON, but a manifest round-tripped
        # through `dataclasses.asdict` keeps `params` a tuple, and both must reload.
        if not isinstance(payload["params"], (list, tuple)) or not payload["params"]:
            raise SynthError("Manifest field 'params' must be a non-empty array")
        params = []
        for entry in payload["params"]:
            if not isinstance(entry, dict):
                raise SynthError(f"Each param must be an object, got {entry!r}")
            if entry.get("type") not in _PARAM_TYPES:
                raise SynthError(
                    f"Param {entry.get('name')!r} has type {entry.get('type')!r}; "
                    f"use one of {sorted(_PARAM_TYPES)}"
                )
            params.append(
                ParamSpec(
                    name=str(entry["name"]),
                    type=str(entry["type"]),
                    minimum=float(entry["minimum"]),
                    maximum=float(entry["maximum"]),
                )
            )
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise SynthError(f"Manifest declares duplicate parameter names: {names}")
        return cls(
            name=str(payload["name"]),
            intent=str(payload["intent"]),
            params=tuple(params),
            source=str(payload["source"]),
        )

    def signature(self) -> str:
        """Render the declared signature the way the prompt and logs show it."""
        args = ", ".join(f"{p.name}: {p.type} [{p.minimum}, {p.maximum}]" for p in self.params)
        return f"{self.name}({args})"

    def bind(self, values: list[Any]) -> tuple[int | float, ...]:
        """Coerce and range-check a positional argument list against the declared parameters.

        Raises:
            SynthError: If the arity is wrong or any value fails its parameter's check.
        """
        if len(values) != self.n_args:
            raise SynthError(
                f"{self.name} takes {self.n_args} arguments {[p.name for p in self.params]}, "
                f"got {len(values)}: {values}"
            )
        return tuple(spec.coerce(value) for spec, value in zip(self.params, values))

    def compile(
        self,
    ) -> tuple[
        Callable[[tuple, NDArray, float, float, dict[str, NDArray]], tuple],
        Callable[[tuple, int], NDArray],
    ]:
        """Compile the shape in the sandbox and wrap it as a primitive.

        Returns:
            The primitive callable, and the bare shape function behind it.
        """
        shape_fn = compile_shape(self.source, self.name)
        return as_primitive(shape_fn), shape_fn

    def register(self, fn: Callable[..., tuple]) -> None:
        """Make the compiled primitive resolvable, emittable, and visible in the prompt."""
        register_synthesized(self.name, fn, self.n_args)
        register_synthesized_action(
            self.name, self.intent, [(p.name, p.type, p.minimum, p.maximum) for p in self.params]
        )

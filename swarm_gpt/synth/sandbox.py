"""Compile and run LLM-authored primitive source under a restricted namespace.

The threat model is buggy generated code on the author's own machine -- a stray file write, an
infinite loop, a NaN -- not an adversary. This is NOT a security boundary: an AST whitelist plus a
restricted ``__builtins__`` is defeatable by a determined attacker, and nothing here should be
exposed to untrusted input.
"""

from __future__ import annotations

import ast
import builtins
import signal
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# The 5-parameter contract every primitive in `motion_primitives.py` obeys.
PRIMITIVE_ARGS = ("params", "swarm_pos", "tstart", "tend", "limits")
INVARIANT_ARGS = ("pos", "time", "params")
INVARIANT_FN_NAME = "check"

_BANNED_NAMES = frozenset(
    {
        "exec",
        "eval",
        "open",
        "compile",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "__import__",
        "memoryview",
    }
)
_ALLOWED_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "divmod",
        "enumerate",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "pow",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "sum",
        "tuple",
        "zip",
        "ValueError",
        "TypeError",
        "IndexError",
        "KeyError",
        "ZeroDivisionError",
    }
)
# Seconds a single primitive or invariant call may run before SIGALRM cuts it off.
_CALL_TIMEOUT_S = 5.0


class SynthError(Exception):
    """A synthesized primitive failed to compile, run, or honour its output contract.

    The message is written to be handed straight back to the LLM as feedback.
    """


def _reject_unsafe(tree: ast.AST) -> None:
    """Walk the parsed source and raise on constructs the sandbox will not run.

    Raises:
        SynthError: On imports, dunder access, or a banned builtin name.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SynthError(
                "Imports are not allowed. numpy is already bound as `np`; write everything else "
                "with builtins."
            )
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SynthError("`global` and `nonlocal` are not allowed in a primitive.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SynthError(f"Attribute access to {node.attr!r} is not allowed.")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise SynthError(f"Access to {node.id!r} is not allowed.")
            if node.id in _BANNED_NAMES:
                raise SynthError(f"{node.id!r} is not available inside a primitive.")


def _sandbox_namespace() -> dict[str, Any]:
    """A fresh execution namespace holding numpy and a whitelisted builtins mapping."""
    allowed = {name: getattr(builtins, name) for name in _ALLOWED_BUILTINS}
    return {"np": np, "__builtins__": allowed}


def _compile_function(
    source: str, name: str, arg_names: tuple[str, ...], field: str
) -> Callable[..., Any]:
    """Parse, vet, and exec ``source``, returning its single top-level function ``name``.

    ``field`` names the manifest field the source came from. Both fields carry Python, so an error
    that does not say which one it came from sends the author looking in the wrong place.

    Raises:
        SynthError: On a syntax error, a rejected construct, a missing or misnamed function, or a
            signature that does not match ``arg_names``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SynthError(
            f"The `{field}` field is not valid Python: {e.__class__.__name__}: {e}. It must hold "
            f"the source of `def {name}({', '.join(arg_names)})`, not a description of it."
        ) from e
    _reject_unsafe(tree)

    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != len(functions) or not functions:
        raise SynthError(f"The `{field}` field must contain only function definitions.")
    target = next((f for f in functions if f.name == name), None)
    if target is None:
        raise SynthError(
            f"The `{field}` field must define a function named {name!r}; it defines "
            f"{[f.name for f in functions]}."
        )
    got = tuple(a.arg for a in target.args.args)
    if got != arg_names:
        raise SynthError(f"{name} in `{field}` must take exactly {arg_names}, got {got}.")

    namespace = _sandbox_namespace()
    try:
        exec(compile(tree, filename=f"<synth:{name}>", mode="exec"), namespace)
    except Exception as e:
        raise SynthError(f"The `{field}` field failed to load: {e.__class__.__name__}: {e}") from e
    return namespace[name]


def compile_primitive(source: str, name: str) -> Callable[..., Any]:
    """Compile LLM-authored primitive source into a callable honouring the primitive contract."""
    return _compile_function(source, name, PRIMITIVE_ARGS, "source")


def compile_invariants(source: str) -> Callable[..., Any]:
    """Compile the LLM's own shape check, a ``check(pos, time, params)`` predicate set."""
    return _compile_function(source, INVARIANT_FN_NAME, INVARIANT_ARGS, "invariants")


def call_guarded(fn: Callable[..., Any], *args: Any) -> Any:
    """Call ``fn`` with a wall-clock guard, converting any failure into feedback.

    Raises:
        SynthError: If the call exceeds the timeout or raises.
    """

    def _timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"exceeded {_CALL_TIMEOUT_S:.0f}s")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, _CALL_TIMEOUT_S)
    try:
        return fn(*args)
    except SynthError:
        raise
    except Exception as e:
        raise SynthError(f"Call raised {e.__class__.__name__}: {e}") from e
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def validate_waypoints(
    result: Any, n_drones: int, tstart: float, tend: float
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Check a primitive's return value against the output contract the pipeline relies on.

    Returns:
        The validated ``(final_pos, waypoints)`` pair, positions in cm.

    Raises:
        SynthError: On any contract violation, phrased as feedback.
    """
    if not isinstance(result, tuple) or len(result) != 2:
        raise SynthError(
            f"A primitive must return (final_pos, waypoints), got {type(result).__name__}."
        )
    final_pos, waypoints = result
    final_pos = np.asarray(final_pos, dtype=float)
    if final_pos.shape != (n_drones, 3):
        raise SynthError(
            f"final_pos must have shape ({n_drones}, 3), got {tuple(final_pos.shape)}."
        )
    if not np.all(np.isfinite(final_pos)):
        raise SynthError("final_pos contains NaN or inf.")
    if not isinstance(waypoints, dict) or not waypoints:
        raise SynthError("waypoints must be a non-empty dict of {time: {drone_id: (3,) array}}.")

    validated: dict[float, dict[int, NDArray]] = {}
    for t, entry in waypoints.items():
        # np.float32 is not a Python float, and `np.linspace` keys are the natural way to emit
        # these, so the numpy scalar types are accepted alongside the builtins.
        if not isinstance(t, (int, float, np.integer, np.floating)) or not np.isfinite(t):
            raise SynthError(f"Waypoint time {t!r} is not a finite number.")
        if not (tstart < t <= tend):
            raise SynthError(
                f"Waypoint time {float(t):.3f} is outside the interval this primitive was given, "
                f"({tstart:.3f}, {tend:.3f}]. Emit every waypoint inside it."
            )
        if not isinstance(entry, dict) or not entry:
            raise SynthError(f"Waypoint at t={float(t):.3f} must be a non-empty {{drone_id: pos}}.")
        positions: dict[int, NDArray] = {}
        for drone_id, pos in entry.items():
            if not isinstance(drone_id, (int, np.integer)) or not 0 <= drone_id < n_drones:
                raise SynthError(
                    f"Waypoint at t={float(t):.3f} names drone {drone_id!r}; drone ids are "
                    f"0-indexed and must be in 0..{n_drones - 1}."
                )
            pos = np.asarray(pos, dtype=float)
            if pos.shape != (3,):
                raise SynthError(
                    f"Position for drone {drone_id} at t={float(t):.3f} must have shape (3,), "
                    f"got {tuple(pos.shape)}."
                )
            if not np.all(np.isfinite(pos)):
                raise SynthError(
                    f"Position for drone {drone_id} at t={float(t):.3f} is not finite."
                )
            positions[int(drone_id)] = pos
        validated[float(t)] = positions
    return final_pos, dict(sorted(validated.items()))

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
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from swarm_gpt.core.motion_primitives import _assign_positions, _formation_arrival_time

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


# The formation helpers every hand-written primitive already calls. Bound by reference rather
# than reimplemented, so an authored primitive gets the library's own assignment and time budget.
HELPERS = {"assign": _assign_positions, "arrival_time": _formation_arrival_time}


def _sandbox_namespace() -> dict[str, Any]:
    """A fresh execution namespace holding numpy, the formation helpers, and safe builtins."""
    allowed = {name: getattr(builtins, name) for name in _ALLOWED_BUILTINS}
    return {"np": np, "__builtins__": allowed, **HELPERS}


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
    if threading.current_thread() is not threading.main_thread():
        return _call_guarded_offthread(fn, *args)

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


def _call_guarded_offthread(fn: Callable[..., Any], *args: Any) -> Any:
    """The same guard for a worker thread, where SIGALRM cannot be installed.

    A runaway call is abandoned rather than interrupted -- Python offers no way to stop another
    thread -- so it keeps burning a daemon thread until the process exits. The loop must stay
    alive, which is what this buys, and the AST whitelist is the guard that actually matters.

    Raises:
        SynthError: If the call exceeds the timeout or raises.
    """
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn(*args)
        except BaseException as e:  # noqa: BLE001 -- re-raised on the calling thread below
            outcome["error"] = e

    worker = threading.Thread(target=_target, daemon=True, name="synth-primitive-call")
    worker.start()
    worker.join(_CALL_TIMEOUT_S)
    if worker.is_alive():
        raise SynthError(f"Call raised TimeoutError: exceeded {_CALL_TIMEOUT_S:.0f}s")
    error = outcome.get("error")
    if isinstance(error, SynthError):
        raise error
    if error is not None:
        raise SynthError(f"Call raised {error.__class__.__name__}: {error}") from error
    return outcome["value"]


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

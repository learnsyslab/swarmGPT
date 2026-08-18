"""The primitive vocabularies a coverage probe can be run against.

Three variants, because the gap depends entirely on which one you mean:

- ``current``   what this working tree's prompt shows the LLM.
- ``sg2``       what ``swarmgpt2-spline-foundation``'s prompt shows the LLM: the same names minus
                ``move`` and ``swap``, which that branch deleted, plus TRANSITION.
- ``sg2_full``  every primitive ``blocks.py`` registers on that branch -- 30 of them, against 12
                exposed. The difference is unwired capability, not missing capability, so measuring
                against it separates a plumbing problem from a synthesis argument.

``sg2_full`` is generated from the source by AST, never hand-transcribed: a hand-written list would
drift from ``blocks.py`` and quietly overstate or understate what exists.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SG2_REF = "swarmgpt2-spline-foundation"
VARIANTS = ("current", "sg2", "sg2_full")


def _git_show(ref: str, path: str) -> str:
    """Read a file from a git ref without touching the working tree."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _block_from_prompt(text: str, tag: str) -> str | None:
    """Pull one tagged block out of a prompt yaml's ``user_initial``, or None if absent."""
    body = yaml.safe_load(text)["user_initial"]
    match = re.search(rf"<{tag}>(.*?)</{tag}>", body, re.DOTALL)
    return match.group(1).strip() if match else None


def _param_names(fn: ast.FunctionDef) -> list[str]:
    """Recover a block primitive's parameter names from its ``x, y = params`` unpacking.

    The spline primitives all take an opaque ``params`` tuple and destructure it in the body, so
    the signature alone says nothing about arity.
    """
    for node in fn.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        value = node.value
        if not (isinstance(value, ast.Name) and value.id == "params"):
            continue
        target = node.targets[0]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [e.id.lstrip("_") for e in target.elts if isinstance(e, ast.Name)]
        if isinstance(target, ast.Name):
            return [target.id.lstrip("_")]
    return []


def sg2_full_block() -> str:
    """Render every primitive ``blocks.py`` registers on the spline branch, as a prompt block."""
    source = _git_show(SG2_REF, "swarm_gpt/core/blocks.py")
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    registered: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", "") == "SPLINE_PRIMITIVES"
        ):
            for key, value in zip(node.value.keys, node.value.values):
                registered.append((key.value, value.id))
            break

    lines, seen = [], set()
    for exposed_name, fn_name in registered:
        fn = functions.get(fn_name)
        if fn is None or exposed_name in seen:
            continue
        seen.add(exposed_name)
        doc = (ast.get_docstring(fn) or "").split("\n")[0].strip()
        names = _param_names(fn)
        # A few unpack `params` in a shape this recovers nothing from. Rendering those as `()`
        # would tell the reader they take no arguments, which is worse than admitting ignorance.
        args = ", ".join(names) if names else "..."
        lines.append(f"  - {exposed_name}({args}) — {doc}")
    return "\n".join(lines)


def vocabulary(variant: str) -> tuple[str, str]:
    """Return the ``(motion, lighting)`` prompt blocks for a named variant.

    Raises:
        ValueError: If ``variant`` is not one of `VARIANTS`.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown vocabulary variant {variant!r}; expected one of {VARIANTS}")
    local = (ROOT / "swarm_gpt/data/motion_primitive_prompts.yaml").read_text()
    # The spline branch predates the lighting primitives entirely, so its prompt has no lighting
    # block. Lighting is orthogonal to the motion rework, so every variant gets this tree's -- a
    # variant with no lighting vocabulary would report a lighting gap that is an artifact of
    # branch age rather than a real limitation.
    lighting = _block_from_prompt(local, "lighting")
    if variant == "current":
        return _block_from_prompt(local, "primitives"), lighting
    if variant == "sg2_full":
        return sg2_full_block(), lighting
    sg2 = _git_show(SG2_REF, "swarm_gpt/data/motion_primitive_prompts.yaml")
    return _block_from_prompt(sg2, "primitives"), lighting


if __name__ == "__main__":
    for name in VARIANTS:
        motion, _ = vocabulary(name)
        count = len(re.findall(r"^\s*-\s+\w+\(", motion, re.MULTILINE))
        print(f"\n{'=' * 70}\n{name}: {count} motion entries\n{'=' * 70}")
        print(motion)

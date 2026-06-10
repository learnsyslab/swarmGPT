"""Batch-analyze every un-analyzed song in ``music/`` via all-in-one.

Run with::

    pixi run -e music analyze

Scans ``music/songs/*.mp3`` (skipping any ``[deploy]`` variants), and for each song that
doesn't already have a JSON under ``music/analyzed/``, runs ``allin1.analyze``, caches the
result, and writes a PNG visualization under ``music/viz/``. Existing JSONs are left
untouched (their visualizations are not regenerated).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import torch
import yaml

from swarm_gpt.utils.music_analyzer import analyze_song

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MUSIC_DIR = PROJECT_ROOT / "music"
SONGS_DIR = MUSIC_DIR / "songs"
ANALYZED_DIR = MUSIC_DIR / "analyzed"
VIZ_DIR = MUSIC_DIR / "viz"
SETTINGS_PATH = PROJECT_ROOT / "swarm_gpt" / "data" / "settings.yaml"


def _ensure_song_crop(song_stem: str) -> None:
    """Add ``song_stem`` to ``song_crops`` in settings.yaml if it is not already present.

    Uses a line-oriented insertion so comments and formatting are fully preserved.
    The new entry is appended after the last existing line in the ``song_crops:`` block.

    Args:
        song_stem: MP3 filename without extension, used as the settings key.
    """
    text = SETTINGS_PATH.read_text()

    # Quick check via YAML parse — avoids a text-insert if the key already exists.
    settings = yaml.safe_load(text)
    if song_stem in settings.get("song_crops", {}):
        return

    # Find the song_crops: block and locate its last indented line.
    lines = text.splitlines(keepends=True)
    in_block = False
    last_entry_idx = -1
    for i, line in enumerate(lines):
        if re.match(r"^song_crops\s*:", line):
            in_block = True
            continue
        if in_block:
            if re.match(r"^\S", line):
                # A non-indented line marks the end of the block.
                break
            if line.strip() and not line.strip().startswith("#"):
                last_entry_idx = i

    if last_entry_idx == -1:
        # Fallback: couldn't find the block — print a warning and skip.
        print(
            f"  WARNING: could not locate song_crops block in {SETTINGS_PATH}; "
            f"add '{song_stem}: [0, 60]' manually."
        )
        return

    new_line = f'  "{song_stem}": [0, 60]\n'
    lines.insert(last_entry_idx + 1, new_line)
    SETTINGS_PATH.write_text("".join(lines))
    print(f"  Added '{song_stem}' to song_crops in settings.yaml with default crop [0, 60].")


def main() -> None:
    """Scan ``music/``, analyze every song that is missing a cached JSON."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no CUDA detected. Analysis on CPU is ~10x+ slower.")
    print(f"PyTorch: {torch.__version__}")
    print(f"Songs dir:    {SONGS_DIR}")
    print(f"Analyzed dir: {ANALYZED_DIR}")
    print(f"Viz dir:      {VIZ_DIR}")
    print()

    mp3s = sorted(p for p in SONGS_DIR.glob("*.mp3") if not p.stem.endswith("[deploy]"))
    if not mp3s:
        print(f"No MP3s found in {SONGS_DIR}.")
        sys.exit(1)

    print(f"Found {len(mp3s)} songs:")
    pending: list[Path] = []
    for mp3 in mp3s:
        cache_path = ANALYZED_DIR / f"{mp3.stem}.json"
        marker = "[cached]" if cache_path.exists() else "[pending]"
        print(f"  {marker} {mp3.stem}")
        if not cache_path.exists():
            pending.append(mp3)
    print()

    if not pending:
        print("All songs already analyzed. Nothing to do.")
        return

    print(f"Analyzing {len(pending)} song(s)...")
    print()

    failures: list[tuple[str, str]] = []
    total_start = time.perf_counter()
    for i, mp3 in enumerate(pending, start=1):
        print(f"[{i}/{len(pending)}] {mp3.stem}")
        t0 = time.perf_counter()
        try:
            structure = analyze_song(mp3, ANALYZED_DIR, device=device, viz_dir=VIZ_DIR)
        except Exception as e:
            print(f"  FAILED: {e.__class__.__name__}: {e}")
            failures.append((mp3.stem, f"{e.__class__.__name__}: {e}"))
            continue
        elapsed = time.perf_counter() - t0
        beats = sum(len(bar.beats) for seg in structure.segments for bar in seg.bars)
        print(
            f"  done in {elapsed:5.1f}s -- "
            f"{structure.bpm} BPM, "
            f"{len(structure.segments)} segments, "
            f"{beats} beats"
        )
        _ensure_song_crop(mp3.stem)

    total_elapsed = time.perf_counter() - total_start
    print()
    succeeded = len(pending) - len(failures)
    print(f"Done. {succeeded}/{len(pending)} succeeded in {total_elapsed:.1f}s total.")
    if failures:
        print(f"{len(failures)} failure(s):")
        for stem, err in failures:
            print(f"  - {stem}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

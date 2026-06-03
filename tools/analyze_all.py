"""Batch-analyze every un-analyzed song in ``music/`` via all-in-one.

Run with::

    pixi run -e music analyze

Scans ``music/*.mp3`` (skipping any ``[deploy]`` variants), and for each song that
doesn't already have a JSON under ``music/analyzed/``, runs ``allin1.analyze`` and
caches the result. Existing JSONs are left untouched.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

from swarm_gpt.utils.music_analyzer import analyze_song

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MUSIC_DIR = PROJECT_ROOT / "music"
ANALYZED_DIR = MUSIC_DIR / "analyzed"


def main() -> None:
    """Scan ``music/``, analyze every song that is missing a cached JSON."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no CUDA detected. Analysis on CPU is ~10x+ slower.")
    print(f"PyTorch: {torch.__version__}")
    print(f"Music dir:    {MUSIC_DIR}")
    print(f"Analyzed dir: {ANALYZED_DIR}")
    print()

    mp3s = sorted(p for p in MUSIC_DIR.glob("*.mp3") if not p.stem.endswith("[deploy]"))
    if not mp3s:
        print(f"No MP3s found in {MUSIC_DIR}.")
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
            structure = analyze_song(mp3, ANALYZED_DIR, device=device)
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

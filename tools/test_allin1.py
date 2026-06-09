"""Smoke test for all-in-one music analysis.

Runs `allin1.analyze` on one MP3 from `music/songs/`, prints the result plus timing and
device info, and saves all-in-one's RMS-over-segments visualization as a PNG under
`music/viz/`. Intended to be run on the Linux GPU box to verify that the pixi install
pulled CUDA-enabled PyTorch + NATTEN + madmom correctly, and to eyeball the analysis
(e.g. full song vs. a snippet by running it on each file in turn).

Usage:
    pixi run -e music python tools/test_allin1.py
    pixi run -e music python tools/test_allin1.py "Walking on Sunshine"
"""

import sys
import time
from pathlib import Path

import allin1
import torch

from swarm_gpt.utils.music_analyzer import save_visualization

DEFAULT_SONG = "On & On"


def main(song_name: str) -> None:
    """Run all-in-one analysis on a single song and print the result.

    Args:
        song_name: Stem of an MP3 in the ``music/`` directory (no extension).
    """
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device:   {torch.cuda.get_device_name(0)}")
        print(f"CUDA version:  {torch.version.cuda}")
    print(f"Analysis device: {device}")
    print()

    music_dir = Path(__file__).resolve().parent.parent / "music" / "songs"
    song_path = music_dir / f"{song_name}.mp3"
    if not song_path.exists():
        print(f"ERROR: song not found at {song_path}")
        print("Available songs:")
        for f in sorted(music_dir.glob("*.mp3")):
            print(f"  - {f.stem}")
        sys.exit(1)
    print(f"Analyzing: {song_path}")
    print()

    t0 = time.perf_counter()
    result = allin1.analyze(str(song_path), device=device)
    elapsed = time.perf_counter() - t0

    if isinstance(result, list):
        result = result[0]

    print(f"Analysis took {elapsed:.2f}s")
    print()
    print("=== Raw result ===")
    print(result)
    print()
    print(f"BPM:           {result.bpm}")
    print(f"Beats:         {len(result.beats)}")
    print(f"Downbeats:     {len(result.downbeats)}")
    print(f"Segments:      {len(result.segments)}")
    print()
    print("Segments:")
    for seg in result.segments:
        print(f"  {seg.start:6.2f}s - {seg.end:6.2f}s : {seg.label}")
    print()
    print("First 20 beats (time, position_in_bar):")
    n = min(20, len(result.beats))
    for i in range(n):
        t = result.beats[i]
        p = result.beat_positions[i] if i < len(result.beat_positions) else None
        print(f"  beat {i + 1:3d}: t={t:7.3f}s  pos_in_bar={p}")

    # Save all-in-one's RMS-over-segments visualization as a PNG under music/viz/.
    viz_dir = Path(__file__).resolve().parent.parent / "music" / "viz"
    out_path = save_visualization(result, viz_dir, song_path.stem)
    print()
    print(f"Visualization saved to: {out_path}")


if __name__ == "__main__":
    song = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SONG
    main(song)

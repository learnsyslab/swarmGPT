"""SongStructure data model and per-song analysis orchestration.

Provides the hierarchical music-structure types the choreographer addresses moments by
(``segment``, ``bar``, ``beat``), plus :func:`analyze_song` which runs all-in-one on an
MP3 and caches the result as JSON.

At runtime the choreographer reads JSONs from ``music/analyzed/`` and never invokes
``allin1.analyze`` itself. allin1 is imported lazily inside :func:`analyze_song` so this
module can be imported in environments (e.g. ``tests``) that do not have allin1.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import allin1
    import librosa
    import matplotlib.pyplot as plt

    from swarm_gpt.core.structured_output_schema import encode_key
except ImportError as e:
    logger.error(f"{e} - please use the music env to analyze songs")

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = 2
"""On-disk JSON schema version. Bump on incompatible shape changes."""

# Match allin1.visualize's librosa params for consistency with its RMS trace.
_FRAME_LENGTH = 4096
_HOP_LENGTH = 1024


@dataclass
class Beat:
    """One beat within a bar.

    Attributes:
        id: 1-indexed beat number within the bar.
        time_s: Time in seconds since song start.
        position_in_bar: Metric position (1 = downbeat, 2/3/4 = off-beats).
    """

    id: int
    time_s: float
    position_in_bar: int


@dataclass
class Bar:
    """One bar (measure) within a segment.

    Attributes:
        id: 1-indexed bar number within the segment.
        start_s: Time in seconds when this bar starts.
        beats: Beats within this bar, in time order.
    """

    id: int
    start_s: float
    beats: list[Beat]


@dataclass
class Segment:
    """One functional segment (intro / verse / chorus / etc.) of the song.

    Attributes:
        id: 1-indexed segment number within the song.
        label: Functional label from all-in-one (e.g. ``"intro"``, ``"chorus"``).
        start_s: Time in seconds when this segment starts.
        end_s: Time in seconds when this segment ends.
        bars: Bars within this segment, in time order.
    """

    id: int
    label: str
    start_s: float
    end_s: float
    bars: list[Bar]


@dataclass
class SongStructure:
    """Hierarchical music structure for a single song.

    Attributes:
        schema_version: Format version of the JSON serialization.
        source_path: Path to the source audio file, as a string relative to project root.
        song_sha256: SHA-256 of the source audio file, used to detect MP3 changes.
        analyzer: Identifier of the analysis engine that produced this structure.
        bpm: Tempo in beats per minute.
        segments: Functional segments of the song, in time order.
    """

    schema_version: int
    source_path: str
    song_sha256: str
    analyzer: str
    bpm: int
    segments: list[Segment]
    rms_per_2bar: tuple[float, ...] = field(default_factory=tuple)
    centroid_per_2bar: tuple[float, ...] = field(default_factory=tuple)

    @classmethod
    def from_allin1(
        cls, result: Any, source_path: str, song_sha256: str, analyzer: str
    ) -> SongStructure:
        """Build a SongStructure from an ``allin1.AnalysisResult``.

        Groups flat ``beats`` / ``beat_positions`` into bars by detecting position
        resets, then assigns bars to segments by the start time of the bar's first beat.

        Args:
            result: An ``allin1.AnalysisResult`` (duck-typed: needs ``bpm``, ``beats``,
                ``beat_positions``, and ``segments`` whose entries expose ``start``,
                ``end``, ``label``).
            source_path: Source audio file path, as a string relative to project root.
            song_sha256: SHA-256 hex digest of the source audio file.
            analyzer: Identifier of the analyzer used (e.g. ``"allin1@1.1.0"``).

        Returns:
            A populated SongStructure.
        """
        bars_flat = _group_beats_into_bars(list(result.beats), list(result.beat_positions))
        segments_out = _drop_empty_segments(
            _assign_bars_to_segments(bars_flat, list(result.segments))
        )
        raw_bpm = result.bpm
        if raw_bpm is None:
            beats_list = list(result.beats)
            if len(beats_list) >= 2:
                intervals = [beats_list[i + 1] - beats_list[i] for i in range(len(beats_list) - 1)]
                median_interval = sorted(intervals)[len(intervals) // 2]
                raw_bpm = 60.0 / median_interval
            else:
                raw_bpm = 0
        return cls(
            schema_version=SCHEMA_VERSION,
            source_path=source_path,
            song_sha256=song_sha256,
            analyzer=analyzer,
            bpm=int(raw_bpm),
            segments=segments_out,
        )

    @classmethod
    def from_json(cls, path: Path) -> SongStructure:
        """Load a SongStructure from its JSON serialization.

        Args:
            path: Path to the JSON file.

        Returns:
            A populated SongStructure.

        Raises:
            ValueError: If the JSON's ``schema_version`` is missing or unsupported.
        """
        data = json.loads(path.read_text())
        version = data["schema_version"]
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version!r} in {path}; "
                f"this code expects {SCHEMA_VERSION}."
            )
        segments = [
            Segment(
                id=int(s["id"]),
                label=str(s["label"]),
                start_s=float(s["start_s"]),
                end_s=float(s["end_s"]),
                bars=[
                    Bar(
                        id=int(b["id"]),
                        start_s=float(b["start_s"]),
                        beats=[
                            Beat(
                                id=int(beat["id"]),
                                time_s=float(beat["time_s"]),
                                position_in_bar=int(beat["position_in_bar"]),
                            )
                            for beat in b["beats"]
                        ],
                    )
                    for b in s["bars"]
                ],
            )
            for s in data["segments"]
        ]
        segments = _drop_empty_segments(segments)
        return cls(
            schema_version=int(data["schema_version"]),
            source_path=str(data["source_path"]),
            song_sha256=str(data["song_sha256"]),
            analyzer=str(data["analyzer"]),
            bpm=int(data["bpm"]),
            segments=segments,
            rms_per_2bar=tuple(float(v) for v in data["rms_per_2bar"]),
            centroid_per_2bar=tuple(float(v) for v in data["centroid_per_2bar"]),
        )

    def to_json(self, path: Path) -> None:
        """Serialize this SongStructure to JSON.

        Args:
            path: Where to write the JSON file. Parent directory must exist.
        """
        path.write_text(json.dumps(asdict(self), indent=2))

    def time_of(self, seq: int, bar: int, beat: int) -> float:
        """Look up the absolute time of a ``(segment, bar, beat)`` address.

        Args:
            seq: 1-indexed segment id.
            bar: 1-indexed bar id within the segment.
            beat: 1-indexed beat id within the bar.

        Returns:
            Time in seconds since song start.

        Raises:
            KeyError: If the ``(seq, bar, beat)`` tuple does not exist.
        """
        for segment in self.segments:
            if segment.id != seq:
                continue
            for bar_obj in segment.bars:
                if bar_obj.id != bar:
                    continue
                for beat_obj in bar_obj.beats:
                    if beat_obj.id == beat:
                        return beat_obj.time_s
        raise KeyError(f"No beat at (seq={seq}, bar={bar}, beat={beat})")

    def required_keys(self, bars_per_required: int = 1) -> list[tuple[int, int, int]]:
        """Return the ``(seq, bar, beat)`` tuples the LLM must emit actions at.

        The downbeat (first beat) of every ``bars_per_required``-th bar, counted from the start
        of each segment. The first bar of every segment is always required (segment openings are
        musically load-bearing); the stride only thins the bars in between. A stride of 1
        requires every bar's downbeat; a stride of 4 requires bars 1, 5, 9, ... within each
        segment. Beats not returned here remain addressable as optional accents.

        Args:
            bars_per_required: Stride between required downbeats within a segment (>= 1).

        Returns:
            List of ``(seq, bar, beat)`` tuples, in time order.

        Raises:
            ValueError: If ``bars_per_required`` is less than 1.
        """
        if bars_per_required < 1:
            raise ValueError(f"bars_per_required must be >= 1, got {bars_per_required}")
        return [
            (segment.id, bar.id, bar.beats[0].id)
            for segment in self.segments
            for i, bar in enumerate(segment.bars)
            if bar.beats and i % bars_per_required == 0
        ]

    def all_keys(self) -> list[tuple[int, int, int]]:
        """Return every addressable ``(seq, bar, beat)`` tuple in the song.

        Returns:
            Tuples in time order.
        """
        return [
            (segment.id, bar.id, beat.id)
            for segment in self.segments
            for bar in segment.bars
            for beat in bar.beats
        ]

    def crop(self, start_s: float, end_s: float) -> SongStructure:
        """Return a copy restricted to the window ``[start_s, end_s]``, rebased to 0.

        Keeps only beats whose time falls within the window, drops bars and segments left
        empty, renumbers all ids contiguously from 1, and shifts every time so the window
        starts at 0:00. The simulator and player both expect a 0-based timeline, so this is
        applied at load time while the on-disk full-song JSON is left untouched.

        Args:
            start_s: Window start in seconds (song-absolute).
            end_s: Window end in seconds (song-absolute).

        Returns:
            A new SongStructure spanning ``[0, end_s - start_s]``.

        Raises:
            ValueError: If ``end_s`` is not greater than ``start_s``.
        """
        if end_s <= start_s:
            raise ValueError(f"crop end ({end_s}) must be greater than start ({start_s})")
        segments_out: list[Segment] = []
        for seg in self.segments:
            bars_out: list[Bar] = []
            for bar in seg.bars:
                beats_in = [b for b in bar.beats if start_s <= b.time_s <= end_s]
                if not beats_in:
                    continue
                bars_out.append(
                    Bar(
                        id=len(bars_out) + 1,
                        start_s=beats_in[0].time_s - start_s,
                        beats=[
                            Beat(id=i, time_s=b.time_s - start_s, position_in_bar=b.position_in_bar)
                            for i, b in enumerate(beats_in, start=1)
                        ],
                    )
                )
            if not bars_out:
                continue
            segments_out.append(
                Segment(
                    id=len(segments_out) + 1,
                    label=seg.label,
                    start_s=max(seg.start_s, start_s) - start_s,
                    end_s=min(seg.end_s, end_s) - start_s,
                    bars=bars_out,
                )
            )
        # Slice dynamics to the windows that survive. A window survives when its first bar has
        # at least one beat in [start_s, end_s]. Count how many original windows are skipped
        # (their first bar's first beat is before start_s), then take surviving_count from there.
        skip = sum(
            1
            for seg in self.segments
            for i, bar in enumerate(seg.bars)
            if i % 2 == 0 and bar.beats and bar.beats[0].time_s < start_s
        )
        surviving_window_count = sum(1 for seg in segments_out for i in range(0, len(seg.bars), 2))
        rms_cropped = self.rms_per_2bar[skip : skip + surviving_window_count]
        centroid_cropped = self.centroid_per_2bar[skip : skip + surviving_window_count]
        return SongStructure(
            schema_version=self.schema_version,
            source_path=self.source_path,
            song_sha256=self.song_sha256,
            analyzer=self.analyzer,
            bpm=self.bpm,
            segments=segments_out,
            rms_per_2bar=rms_cropped,
            centroid_per_2bar=centroid_cropped,
        )


def compute_dynamics_per_2bar(
    structure: SongStructure, mp3_path: Path
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-2-bar RMS amplitude and spectral centroid, normalized to [0, 1].

    Computes two audio features for each 2-bar window in the song. Windows reset at segment
    boundaries so each window cleanly belongs to one segment. Both features are normalized by
    the song-wide maximum so values are scale-invariant across songs.

    Args:
        structure: The full-song SongStructure (pre-crop). Bars in ``structure.segments``
            define the window boundaries.
        mp3_path: Path to the source audio file.

    Returns:
        ``(rms_tuple, centroid_tuple)`` of equal length
        ``sum(ceil(len(seg.bars) / 2) for seg in structure.segments)``.
    """
    windows: list[tuple[float, float]] = []
    for seg in structure.segments:
        bars = seg.bars
        for i in range(0, len(bars), 2):
            start_bar = bars[i]
            end_bar = bars[i + 1] if i + 1 < len(bars) else bars[i]
            t_start = start_bar.beats[0].time_s if start_bar.beats else start_bar.start_s
            t_end = end_bar.beats[-1].time_s if end_bar.beats else end_bar.start_s
            windows.append((t_start, t_end))
    if not windows:
        return ((), ())

    y, sr = librosa.load(str(mp3_path), sr=None, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=_FRAME_LENGTH, hop_length=_HOP_LENGTH
    )[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=_HOP_LENGTH)

    rms_vals: list[float] = []
    centroid_vals: list[float] = []
    for t_start, t_end in windows:
        mask = (frame_times >= t_start) & (frame_times < t_end)
        rms_vals.append(float(rms[mask].mean()) if mask.any() else 0.0)
        centroid_vals.append(float(centroid[mask].mean()) if mask.any() else 0.0)

    rms_peak = max(rms_vals) or 1.0
    centroid_peak = max(centroid_vals) or 1.0
    return (
        tuple(round(v / rms_peak, 2) for v in rms_vals),
        tuple(round(v / centroid_peak, 2) for v in centroid_vals),
    )


def dynamics_window_keys(structure: SongStructure) -> tuple[str, ...]:
    """Generate the s#b#t# key for the start of each 2-bar window, in order.

    Pure function of the bar layout — not persisted; regenerated whenever the prompt is
    built. Length matches ``structure.rms_per_2bar``.

    Args:
        structure: The SongStructure whose bar layout defines the windows.

    Returns:
        Tuple of key strings, one per 2-bar window.
    """
    keys: list[str] = []
    for seg in structure.segments:
        bars = seg.bars
        for i in range(0, len(bars), 2):
            start_bar = bars[i]
            if start_bar.beats:
                keys.append(encode_key(seg.id, start_bar.id, start_bar.beats[0].id))
    return tuple(keys)


def _group_beats_into_bars(
    beats: list[float], positions: list[int]
) -> list[list[tuple[float, int]]]:
    """Group flat ``(beat_time, position_in_bar)`` pairs into bars.

    A new bar starts whenever ``position_in_bar`` does not strictly increase relative to
    the previous beat. Handles variable meters (3/4, 4/4, etc.) and anacrusis (a partial
    first bar) implicitly.

    Args:
        beats: Beat times in seconds, in time order.
        positions: Position-in-bar for each beat (1-indexed), parallel to ``beats``.

    Returns:
        List of bars, where each bar is a list of ``(time, position)`` pairs.
    """
    bars: list[list[tuple[float, int]]] = []
    current: list[tuple[float, int]] = []
    for time, pos in zip(beats, positions, strict=True):
        if current and pos <= current[-1][1]:
            bars.append(current)
            current = []
        current.append((float(time), int(pos)))
    if current:
        bars.append(current)
    return bars


def _assign_bars_to_segments(
    bars_flat: list[list[tuple[float, int]]], segments_in: list[Any]
) -> list[Segment]:
    """Assign each flat bar to the segment containing its first beat.

    Args:
        bars_flat: Bars as produced by :func:`_group_beats_into_bars`.
        segments_in: Segments from an ``allin1.AnalysisResult`` (duck-typed: ``start``,
            ``end``, ``label``).

    Returns:
        Segments populated with their child bars, in the same order as ``segments_in``.
    """
    segments_out: list[Segment] = []
    bar_cursor = 0
    for seg_id, raw_seg in enumerate(segments_in, start=1):
        seg_bars: list[Bar] = []
        bar_id = 0
        while bar_cursor < len(bars_flat):
            bar_beats = bars_flat[bar_cursor]
            bar_start_time = bar_beats[0][0]
            if bar_start_time >= float(raw_seg.end):
                break
            bar_id += 1
            seg_bars.append(
                Bar(
                    id=bar_id,
                    start_s=float(bar_beats[0][0]),
                    beats=[
                        Beat(id=beat_id, time_s=t, position_in_bar=p)
                        for beat_id, (t, p) in enumerate(bar_beats, start=1)
                    ],
                )
            )
            bar_cursor += 1
        segments_out.append(
            Segment(
                id=seg_id,
                label=str(raw_seg.label),
                start_s=float(raw_seg.start),
                end_s=float(raw_seg.end),
                bars=seg_bars,
            )
        )
    return segments_out


def _drop_empty_segments(segments: list[Segment]) -> list[Segment]:
    """Drop segments that contain no beats and renumber the survivors from 1.

    allin1 can emit segment boundaries that no beat falls within (e.g. very short
    sections). Such segments carry nothing to choreograph and would only mislead the LLM,
    so they are removed and the remaining segment ids are made contiguous.

    Args:
        segments: Segments in time order, possibly including empty ones.

    Returns:
        The non-empty segments with ``id`` renumbered to be contiguous from 1.
    """
    kept = [seg for seg in segments if seg.bars]
    for new_id, seg in enumerate(kept, start=1):
        seg.id = new_id
    return kept


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_visualization(result: Any, viz_dir: Path, stem: str) -> Path:
    """Save all-in-one's RMS-over-segments visualization as a PNG.

    Renders the same figure as ``allin1.visualize`` (which only writes PDFs) but saves it
    as a raster image instead.

    Args:
        result: An ``allin1.AnalysisResult``.
        viz_dir: Directory to write the PNG into. Created if missing.
        stem: Output file stem (the song's MP3 stem, no extension).

    Returns:
        Path to the written PNG.
    """
    fig = allin1.visualize(result, out_dir=None, multiprocess=False)
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analyze_song(
    mp3_path: Path, cache_dir: Path, device: str = "cuda", viz_dir: Path | None = None
) -> SongStructure:
    """Analyze a single MP3 and cache the result as JSON.

    If ``cache_dir/<song_stem>.json`` already exists, loads and returns it without
    re-running analysis (and without regenerating any visualization).

    Args:
        mp3_path: Path to the MP3 file.
        cache_dir: Directory to read/write JSON caches in. Created if missing.
        device: Torch device for analysis (``"cuda"`` or ``"cpu"``).
        viz_dir: If given, save a PNG visualization of the analysis here when the song is
            analyzed (skipped on a cache hit). Created if missing.

    Returns:
        The SongStructure for the song.
    """
    cache_path = cache_dir / f"{mp3_path.stem}.json"
    if cache_path.exists():
        return SongStructure.from_json(cache_path)

    result = allin1.analyze(str(mp3_path), device=device)
    if isinstance(result, list):
        result = result[0]

    source_path = _project_relative(mp3_path)
    structure = SongStructure.from_allin1(
        result=result,
        source_path=source_path,
        song_sha256=sha256_of(mp3_path),
        analyzer=f"allin1@{getattr(allin1, '__version__', 'unknown')}",
    )
    rms_per_2bar, centroid_per_2bar = compute_dynamics_per_2bar(structure, mp3_path)
    structure.rms_per_2bar = rms_per_2bar
    structure.centroid_per_2bar = centroid_per_2bar
    cache_dir.mkdir(parents=True, exist_ok=True)
    structure.to_json(cache_path)
    if viz_dir is not None:
        save_visualization(result, viz_dir, mp3_path.stem)
    return structure


def _project_relative(path: Path) -> str:
    """Format a path as a string relative to the project root when possible."""
    resolved = path.resolve()
    # music_analyzer.py lives at <root>/swarm_gpt/utils/music_analyzer.py; go up 3.
    project_root = type(path)(__file__).resolve().parents[2]
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)

"""Unit tests for :mod:`swarm_gpt.utils.music_analyzer`.

Tests use duck-typed synthetic ``AnalysisResult`` objects so allin1 is never imported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from swarm_gpt.utils.music_analyzer import (
    SCHEMA_VERSION,
    Bar,
    Beat,
    Segment,
    SongStructure,
    compute_dynamics_per_2bar,
    dynamics_window_keys,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class FakeAllin1Segment:
    start: float
    end: float
    label: str


@dataclass
class FakeAllin1Result:
    bpm: int
    beats: list[float]
    beat_positions: list[int]
    segments: list[FakeAllin1Segment]


def _three_segment_song_4_4() -> FakeAllin1Result:
    """120 BPM, 4/4 throughout: 3 segments, 1 bar each, 4 beats each."""
    return FakeAllin1Result(
        bpm=120,
        beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        beat_positions=[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        segments=[
            FakeAllin1Segment(start=0.0, end=2.0, label="intro"),
            FakeAllin1Segment(start=2.0, end=4.0, label="verse"),
            FakeAllin1Segment(start=4.0, end=6.0, label="outro"),
        ],
    )


def _waltz_song_3_4() -> FakeAllin1Result:
    """3/4 (waltz): one segment, two bars, three beats each."""
    return FakeAllin1Result(
        bpm=120,
        beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        beat_positions=[1, 2, 3, 1, 2, 3],
        segments=[FakeAllin1Segment(start=0.0, end=3.0, label="whole")],
    )


def _build(result: FakeAllin1Result) -> SongStructure:
    return SongStructure.from_allin1(
        result=result,
        source_path="music/Test.mp3",
        source_sha256="deadbeef",
        analyzer="allin1@test",
    )


def test_from_allin1_groups_beats_into_4_4_bars() -> None:
    structure = _build(_three_segment_song_4_4())
    assert len(structure.segments) == 3
    for seg in structure.segments:
        assert len(seg.bars) == 1
        assert [b.position_in_bar for b in seg.bars[0].beats] == [1, 2, 3, 4]


def test_segment_labels_and_ids_preserved() -> None:
    structure = _build(_three_segment_song_4_4())
    assert [s.label for s in structure.segments] == ["intro", "verse", "outro"]
    assert [s.id for s in structure.segments] == [1, 2, 3]


def test_time_of_returns_beat_time() -> None:
    structure = _build(_three_segment_song_4_4())
    assert structure.time_of(1, 1, 1) == 0.0
    assert structure.time_of(2, 1, 1) == 2.0
    assert structure.time_of(3, 1, 4) == 5.5


def test_time_of_raises_for_missing_key() -> None:
    structure = _build(_three_segment_song_4_4())
    with pytest.raises(KeyError):
        structure.time_of(99, 1, 1)


def test_required_keys_are_bar_downbeats() -> None:
    # One bar per segment here, so the downbeats coincide with the segment openings.
    structure = _build(_three_segment_song_4_4())
    assert structure.required_keys() == [(1, 1, 1), (2, 1, 1), (3, 1, 1)]


def test_required_keys_cover_every_bar() -> None:
    # Two bars in one segment: both bar downbeats are required, not just the segment opening.
    structure = _build(_waltz_song_3_4())
    assert structure.required_keys() == [(1, 1, 1), (1, 2, 1)]


def test_all_keys_covers_every_beat() -> None:
    structure = _build(_three_segment_song_4_4())
    # 3 segments x 1 bar x 4 beats = 12
    assert len(structure.all_keys()) == 12


def test_variable_meter_3_4() -> None:
    structure = _build(_waltz_song_3_4())
    assert len(structure.segments) == 1
    assert len(structure.segments[0].bars) == 2
    assert all(len(bar.beats) == 3 for bar in structure.segments[0].bars)


def test_json_round_trip(tmp_path: Path) -> None:
    original = _build(_three_segment_song_4_4())
    path = tmp_path / "test.json"
    original.to_json(path)
    loaded = SongStructure.from_json(path)
    assert loaded == original


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(ValueError, match="schema_version"):
        SongStructure.from_json(path)


def test_schema_version_constant_is_current() -> None:
    # Guards against accidentally bumping SCHEMA_VERSION without updating tests/JSONs.
    assert SCHEMA_VERSION == 2


def test_crop_selects_window_and_rebases_to_zero() -> None:
    structure = _build(_three_segment_song_4_4())  # 3 segments, beats 0.0..5.5
    cropped = structure.crop(2.0, 3.9)  # only the "verse" segment's beats fall inside
    assert len(cropped.segments) == 1
    seg = cropped.segments[0]
    assert seg.label == "verse"
    assert seg.id == 1
    assert seg.start_s == 0.0  # rebased: window start -> 0
    assert len(seg.bars) == 1
    beats = seg.bars[0].beats
    assert [b.time_s for b in beats] == [0.0, 0.5, 1.0, 1.5]  # 2.0..3.5 shifted by -2.0
    assert [b.position_in_bar for b in beats] == [1, 2, 3, 4]
    assert [b.id for b in beats] == [1, 2, 3, 4]


def test_crop_renumbers_ids_contiguously() -> None:
    structure = _build(_three_segment_song_4_4())
    cropped = structure.crop(2.0, 6.0)  # drops the intro, keeps verse + outro
    assert [seg.id for seg in cropped.segments] == [1, 2]
    assert [seg.label for seg in cropped.segments] == ["verse", "outro"]
    assert cropped.segments[0].bars[0].beats[0].time_s == 0.0


def test_crop_full_window_preserves_keys() -> None:
    structure = _build(_three_segment_song_4_4())
    cropped = structure.crop(0.0, 6.0)
    assert cropped.all_keys() == structure.all_keys()
    assert cropped.segments[0].bars[0].beats[0].time_s == 0.0


def test_crop_rejects_non_positive_window() -> None:
    structure = _build(_three_segment_song_4_4())
    with pytest.raises(ValueError, match="must be greater"):
        structure.crop(3.0, 3.0)
    with pytest.raises(ValueError, match="must be greater"):
        structure.crop(4.0, 2.0)


# ---------------------------------------------------------------------------
# F2 helpers
# ---------------------------------------------------------------------------


def _make_synthetic_structure(n_bars: int, bar_dur: float) -> SongStructure:
    """Single-segment structure with ``n_bars`` bars of ``bar_dur`` seconds each."""
    bars: list[Bar] = []
    t = 0.0
    for bar_id in range(1, n_bars + 1):
        beats = [Beat(id=1, time_s=t, position_in_bar=1)]
        bars.append(Bar(id=bar_id, start_s=t, beats=beats))
        t += bar_dur
    seg = Segment(id=1, label="test", start_s=0.0, end_s=t, bars=bars)
    return SongStructure(
        schema_version=SCHEMA_VERSION,
        source_path="test.mp3",
        source_sha256="abc",
        analyzer="test",
        bpm=120,
        segments=[seg],
    )


def _make_synthetic_structure_two_segments(bars_per_seg: int) -> SongStructure:
    """Two-segment structure, each with ``bars_per_seg`` bars at 1s each."""
    bar_dur = 1.0
    segments: list[Segment] = []
    t = 0.0
    for seg_id in range(1, 3):
        bars: list[Bar] = []
        seg_start = t
        for bar_id in range(1, bars_per_seg + 1):
            beats = [Beat(id=1, time_s=t, position_in_bar=1)]
            bars.append(Bar(id=bar_id, start_s=t, beats=beats))
            t += bar_dur
        segments.append(
            Segment(id=seg_id, label=f"seg{seg_id}", start_s=seg_start, end_s=t, bars=bars)
        )
    return SongStructure(
        schema_version=SCHEMA_VERSION,
        source_path="test.mp3",
        source_sha256="abc",
        analyzer="test",
        bpm=120,
        segments=segments,
    )


def test_rms_per_2bar_aligns_with_bars(tmp_path: "Path") -> None:
    """Bars 1-2 silent, bars 3-4 tone → first window quiet, second loud."""
    import soundfile as sf

    sr = 22050
    bar_dur = 1.0
    n_bars = 4
    y = np.zeros(int(n_bars * bar_dur * sr))
    y[int(2 * bar_dur * sr) :] = 0.5 * np.sin(
        2 * np.pi * 440 * np.arange(len(y) - int(2 * bar_dur * sr)) / sr
    )
    wav_path = tmp_path / "test.wav"
    sf.write(wav_path, y, sr)
    structure = _make_synthetic_structure(n_bars=4, bar_dur=bar_dur)
    rms, _centroid = compute_dynamics_per_2bar(structure, wav_path)
    assert len(rms) == 2
    assert rms[0] < 0.1, f"silent window should be ~0, got {rms[0]}"
    assert rms[1] > 0.5, f"tone window should be loud, got {rms[1]}"


def test_centroid_distinguishes_low_vs_high_frequencies(tmp_path: "Path") -> None:
    """Bars 1-2 low tone (110 Hz), bars 3-4 high tone (4400 Hz) → centroid rises."""
    import soundfile as sf

    sr = 22050
    bar_dur = 1.0
    n_samples = int(2 * bar_dur * sr)
    t = np.arange(n_samples) / sr
    y_low = 0.5 * np.sin(2 * np.pi * 110 * t)
    y_high = 0.5 * np.sin(2 * np.pi * 4400 * t)
    y = np.concatenate([y_low, y_high])
    wav_path = tmp_path / "test.wav"
    sf.write(wav_path, y, sr)
    structure = _make_synthetic_structure(n_bars=4, bar_dur=bar_dur)
    _rms, centroid = compute_dynamics_per_2bar(structure, wav_path)
    assert centroid[1] > centroid[0] * 2, (
        f"high-frequency window should have ~2x+ centroid, got {centroid}"
    )


def test_dynamics_window_keys_do_not_cross_segment_boundary() -> None:
    """A 3-bar segment 1 + 3-bar segment 2 → 4 keys, each segment's keys stay in that segment."""
    structure = _make_synthetic_structure_two_segments(bars_per_seg=3)
    keys = dynamics_window_keys(structure)
    # Each 3-bar segment: 1 full + 1 truncated window = 2 windows. Total = 4.
    assert len(keys) == 4
    assert keys[0].startswith("s1b") and keys[1].startswith("s1b")
    assert keys[2].startswith("s2b") and keys[3].startswith("s2b")


def test_crop_slices_dynamics_arrays() -> None:
    """crop() trims rms_per_2bar / centroid_per_2bar to windows that survive."""
    structure = _make_synthetic_structure(n_bars=8, bar_dur=1.0)
    structure.rms_per_2bar = (0.1, 0.2, 0.3, 0.4)
    structure.centroid_per_2bar = (0.5, 0.6, 0.7, 0.8)
    cropped = structure.crop(start_s=2.0, end_s=8.0)
    assert len(cropped.rms_per_2bar) == 3
    assert cropped.rms_per_2bar == (0.2, 0.3, 0.4)
    assert cropped.centroid_per_2bar == (0.6, 0.7, 0.8)


def test_from_json_rejects_v1_cache(tmp_path: "Path") -> None:
    """Loading a schema-v1 JSON raises ValueError after the version bump."""
    v1_path = tmp_path / "old.json"
    v1_path.write_text('{"schema_version": 1, "bpm": 120, "segments": []}')
    with pytest.raises(ValueError, match="schema_version"):
        SongStructure.from_json(v1_path)

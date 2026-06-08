"""Unit tests for :mod:`swarm_gpt.utils.music_analyzer`.

Tests use duck-typed synthetic ``AnalysisResult`` objects so allin1 is never imported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from swarm_gpt.utils.music_analyzer import SCHEMA_VERSION, SongStructure

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
    assert SCHEMA_VERSION == 1

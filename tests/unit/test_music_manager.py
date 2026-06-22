from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from swarm_gpt.utils.music_manager import MusicManager


@pytest.fixture
def music_dir(tmp_path: Path) -> Path:
    song_path = tmp_path / "Example Song.mp3"
    song_path.write_bytes(b"fake mp3")
    return tmp_path


def test_play_applies_crop_window(music_dir: Path) -> None:
    manager = MusicManager(music_dir)
    manager.song = "Example Song"

    media = MagicMock()
    player = MagicMock()
    player.play.return_value = 0
    player.is_playing.return_value = True
    instance = MagicMock()
    instance.media_new.return_value = media
    instance.media_player_new.return_value = player

    with patch.object(manager, "_get_vlc_instance", return_value=instance):
        started = manager.play(wait=True, start_s=12.5, end_s=72.5)

    assert started is True
    media.add_option.assert_any_call(":start-time=12.5")
    media.add_option.assert_any_call(":stop-time=72.5")
    player.set_time.assert_called_once_with(12500)


def test_play_without_crop_window_skips_media_options(music_dir: Path) -> None:
    manager = MusicManager(music_dir)
    manager.song = "Example Song"

    media = MagicMock()
    player = MagicMock()
    player.play.return_value = 0
    instance = MagicMock()
    instance.media_new.return_value = media
    instance.media_player_new.return_value = player

    with patch.object(manager, "_get_vlc_instance", return_value=instance):
        manager.play(wait=False)

    media.add_option.assert_not_called()
    player.set_time.assert_not_called()

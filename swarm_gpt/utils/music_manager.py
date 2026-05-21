"""Module for handling the music playback and beat extraction."""

from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING

import libfmp.c5
import libfmp.c6
import librosa
import matplotlib.pyplot as plt
import numpy as np
import vlc
from mutagen.mp3 import MP3
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class MusicManager:
    """The music manager is responsible for extracting song information and playing the music."""

    min_beat_time: float = 2.0  # Minimum time between beats in seconds

    def __init__(self, music_dir: Path):
        """Read in all available songs from the music directory.

        Args:
            music_dir: Path to the music directory.
        """
        self.music_dir = music_dir
        self.songs = [f.stem for f in music_dir.glob("*.mp3") if not f.stem.endswith("[deploy]")]
        assert not any("|" in s for s in self.songs), "Songs cannot contain |"
        assert len(self.songs) > 0, "No songs found in music directory"
        self._song = ""
        self._vlc_instance: vlc.Instance | None = None
        self._music_player: vlc.MediaPlayer | None = None

    def _vlc_instance_args(self) -> tuple[str, ...]:
        """CLI args for libvlc: one shared instance avoids duplicate Core Audio listeners (macOS)."""
        base: tuple[str, ...] = (
            "--intf=dummy",
            "--no-video",
            "--quiet",
        )
        if sys.platform != "darwin":
            return base
        return (*base, "--aout=audiounit")

    def _get_vlc_instance(self) -> vlc.Instance:
        if self._vlc_instance is None:
            variants: list[tuple[str, ...]] = [self._vlc_instance_args()]
            # Older VLC.app builds may not ship ``audiounit``; fall back to default (usually auhal).
            if sys.platform == "darwin" and any(x == "--aout=audiounit" for x in variants[0]):
                variants.append(("--intf=dummy", "--no-video", "--quiet"))
            last_err: BaseException | None = None
            for args in variants:
                try:
                    inst = vlc.Instance(*args)
                except (AttributeError, OSError, TypeError) as e:
                    last_err = e
                    continue
                if inst is not None:
                    self._vlc_instance = inst
                    return inst
            raise RuntimeError(
                "Could not initialize libvlc; install VLC.app and ensure python-vlc matches it."
            ) from last_err
        return self._vlc_instance

    @property
    def song(self) -> str:
        """Get the song to choreograph."""
        return self._song

    @song.setter
    def song(self, song: str):
        """Set the song to choreograph.

        Args:
            song: The song to choreograph. This song needs to be present in the music directory.
        """
        assert (self.music_dir / (song + ".mp3")).is_file(), "Song not found in music dir"
        self._song = song

    @property
    def song_length(self) -> float:
        """Get the length of the song in seconds."""
        assert self.song, "Song has not been set yet!"
        return MP3(self.music_dir / (self.song + ".mp3")).info.length  # in seconds

    def play(self, *, wait: bool = False, timeout: float = 2.0) -> bool:
        """Play the song with VLC.

        Args:
            wait: If True, block briefly until VLC reports active playback.
            timeout: Maximum time to wait for playback to start.

        Returns:
            True if VLC accepted the play request and, when ``wait`` is set, started playback.
        """
        assert self.song, "Song not set"
        media_path = str(self.music_dir / (self.song + ".mp3"))
        inst = self._get_vlc_instance()
        if self._music_player is not None:
            self._music_player.stop()
            self._music_player.release()
            self._music_player = None
        self._music_player = inst.media_player_new()
        self._music_player.set_media(inst.media_new(media_path))
        self._music_player.audio_set_volume(100)
        result = self._music_player.play()
        if result == -1:
            logger.error("VLC failed to start playback for %s", media_path)
            return False
        if not wait:
            return True

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.is_playing:
                return True
            time.sleep(0.01)
        logger.warning("Timed out waiting for VLC playback to start for %s", media_path)
        return self.is_playing

    def stop(self):
        """Stop the song."""
        if self._music_player is not None:
            self._music_player.stop()
            self._music_player.release()
            self._music_player = None

    @property
    def is_playing(self) -> bool:
        """Check if the song is playing."""
        if self._music_player is None:
            return False
        return self._music_player.is_playing()

    @property
    def current_time(self) -> float | None:
        """Current VLC playback time in seconds."""
        if self._music_player is None:
            return None
        current_ms = self._music_player.get_time()
        if current_ms < 0:
            return None
        return current_ms / 1000

    def extract_song_info(self) -> dict:
        """Extract the song information."""
        assert self.song, "Song not set"
        nov, fs_nov = self.spectral_novelty(self.song)
        peak_idx = self._peak_detection(nov, fs_nov)
        chords = self.chord_analysis(self.song)
        music_info = {
            "beat_times": np.linspace(0, self.song_length, len(nov))[peak_idx],
            "chords": [chords[i] for i in peak_idx],
            "novelty": nov[peak_idx],
            "dBFS": self.dbfs()[peak_idx],
        }
        return music_info

    def dbfs(self) -> np.ndarray:
        """Compute the dBFS of the song.

        Returns:
            The dBFS of the song.
        """
        # RMS energy
        path = self.music_dir / (self.song + ".mp3")
        assert path.exists(), "Could not find the song in the music directory"
        wav, sr = librosa.load(path)
        N = max(int(0.2 * sr), 1)  # 200ms window size
        H = max(int(0.02 * sr), 1)  # 20ms hop size
        rms = librosa.feature.rms(y=wav, frame_length=N, hop_length=H)[0]
        # Convert to dBFS
        return 20 * np.log10(np.abs(rms) + np.finfo(float).eps)

    def spectral_novelty(self, song: str) -> tuple[np.ndarray, int]:
        """Compute the novelty of the song using a spectral approach.

        See https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_OnsetDetection.html for more
        information.

        Note:
            This function calls the libfmp library, which uses numba to jit compile its code. This
            may cause the first call to this function to take longer for each session. Subsequent
            calls in the same session are faster.

        Args:
            song: The song to compute the novelty for.

        Returns:
            The novelty of the song and the sample rate.
        """
        path = self.music_dir / (song + ".mp3")
        assert path.exists(), "Could not find the song in the music directory"
        wav, sr = librosa.load(path)
        # For parameter meanings, see libfmp docs
        N = max(int(0.2 * sr), 1)  # 200ms window size
        H = max(int(0.02 * sr), 1)  # 20ms hop size
        gamma = 100  # Log smoothing factor
        M = max(int(0.005 * sr), 1)  # 5ms local average window
        nov, fs_nov = libfmp.c6.compute_novelty_spectrum(wav, Fs=sr, N=N, H=H, gamma=gamma, M=M)
        nov[: int(0.2 * fs_nov)] = 0  # Remove the first 200ms because of edge effects
        nov[-int(0.2 * fs_nov) :] = 0  # And the last 200ms
        nov /= np.max(nov)  # Renormalize the novelty function after removing the edges
        return nov, fs_nov

    def _peak_detection(self, nov: np.ndarray, fs_nov: int) -> np.ndarray:
        """Find the peaks in the novelty of the song for musical onset detection.

        See https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_PeakPicking.html for more
        information.

        Args:
            nov: The novelty function of the song.
            fs_nov: The sample rate of the novelty function.

        Returns:
            The indices of the peaks in the novelty function.
        """
        distance = self.min_beat_time * fs_nov  # minimum distance between peaks
        peak_idx, _ = find_peaks(
            nov, height=0.1, distance=distance, prominence=np.percentile(nov, 0.75)
        )
        return peak_idx

    def chord_analysis(self, song: str, plot: bool = False) -> list[str]:
        """Perform chord analysis on the song.

        See https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_ChordRec_HMM.html

        Note:
            This function calls the libfmp library, which uses numba to jit compile its code. This
            may cause the first call to this function to take longer for each session. Subsequent
            calls in the same session are faster.

        Args:
            song: The song to perform the chord analysis on.
            plot: Whether to plot the chords or not.

        Returns:
            The chords of the song.
        """
        path = self.music_dir / (song + ".mp3")
        assert path.exists(), "Could not find the song in the music directory"
        wav, sr = librosa.load(path)
        # Create chroma Short-Time Fourier Transform features
        N = max(int(0.2 * sr), 1)  # 0.2 seconds
        H = max(int(0.02 * sr), 1)
        chords = librosa.feature.chroma_stft(
            y=wav, sr=sr, tuning=0, norm=None, hop_length=H, n_fft=N
        )
        chord_sim, _ = libfmp.c5.chord_recognition_template(chords)  # 24, 12 major and 12 minor
        A = libfmp.c5.uniform_transition_matrix(p=0.5)  # Very simple transition matrix
        C = 1 / 24 * np.ones((1, 24))
        chord_HMM, _, _, _ = libfmp.c5.viterbi_log_likelihood(A, C, chord_sim)
        chord_labels = libfmp.c5.get_chord_labels()
        if plot:  # Plot the chromagram
            librosa.display.specshow(
                10 * np.log10(chords + np.finfo(float).eps),
                x_axis="time",
                y_axis="chroma",
                sr=sr,
                hop_length=H,
            )
            plt.colorbar()
            plt.show()
        return [chord_labels[c] for c in np.argmax(chord_HMM, axis=0)]

    def animate_peaks(self):
        """Play the song, plot its novelty peaks and animate the current time as a moving line."""
        assert self.song, "Song not set"
        # Detect peaks
        nov, fs_nov = self.spectral_novelty(self.song)
        peak_idx = self._peak_detection(nov, fs_nov)
        # Create plot
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(13, 5))
        t_nov = np.linspace(0, self.song_length, len(nov))
        ax.plot(t_nov, nov, label="Novelty function")
        dbfs = self.dbfs()
        # Normalize the dBFS to ~[0, 1] (use 5th percentile as lower bound)
        norm_dbfs = (dbfs - np.quantile(dbfs, 0.05)) / (-np.quantile(dbfs, 0.05))
        ax.plot(t_nov, norm_dbfs, label="dBFS [0, 1] normalized")
        ax.set_ylim(0, 1)
        ax.scatter(t_nov[peak_idx], nov[peak_idx], c="r", label="Novelty peaks")
        ax.legend()
        t_bar = ax.plot([0, 0], [0, 1], c="b")
        # Play the song, animate current time as blue line on the novelty plot
        self.play()
        while not self.is_playing:
            time.sleep(0.001)  # Wait for the player to start
        start_time = time.perf_counter()
        while self.is_playing:
            dt = time.perf_counter() - start_time
            t_bar[0].set_xdata([dt, dt])
            fig.canvas.draw()
            fig.canvas.flush_events()
            # Reduce loop iterations
            time.sleep(0.001)

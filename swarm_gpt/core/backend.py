"""Backend module for the swarm_gpt web app."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, ParamSpec, TypeVar

import numpy as np
import yaml
from scipy.interpolate import make_smoothing_spline

from swarm_gpt.core import Choreographer
from swarm_gpt.core.lighting import compile_cues, load_lighting_config
from swarm_gpt.core.sim import replay_sim_states, simulate_axswarm
from swarm_gpt.exception import LLMException
from swarm_gpt.utils import MusicManager
from swarm_gpt.utils.music_analyzer import SongStructure

if TYPE_CHECKING:
    from numpy.typing import NDArray as Array

    from swarm_gpt.core.drone_swarm import DroneSwarm
    from swarm_gpt.core.lighting import LightingTimeline
    from swarm_gpt.utils.llm_providers import LLMProvider

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

P = ParamSpec("P")  # Represents arbitrary parameters
R = TypeVar("R")  # Represents the return type


def self_correct(n_retries: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Create a decorator that retries a function n times if it fails.

    Args:
        n_retries: Number of times to retry the function
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        """Decorator that retries a function n times if it fails."""

        @wraps(fn)
        def wrapper(self: AppBackend, *args: P.args, **kwargs: P.kwargs) -> R:
            assert isinstance(self, AppBackend), "self_correct decorator must be used on AppBackend"
            try:
                return fn(self, *args, **kwargs)
            except LLMException as e:
                error_message = str(e)
                for i in range(n_retries):
                    try:
                        logger.info("Reprompting due to LLM error")
                        message = "The provided response failed with the following error:"
                        message += f"\n{error_message}\n\n"
                        message += "Analyze the error, re-read the instructions and try again."
                        # Use the underlying, undecorated reprompt function to avoid infinite
                        # recursion.
                        return self.reprompt.__wrapped__(self, message)
                    except LLMException as inner_e:
                        if i == n_retries - 1:
                            raise inner_e
                        error_message = str(inner_e)
                        continue
                raise e

        return wrapper

    return decorator


def _fold_cues_to_rgb(cues: dict[float, Array]) -> dict[str, list]:
    """Fold one drone-deck's WRGB cue dict into the browser's parallel arrays.

    Three.js has no white channel, so W folds into all three exactly as
    :meth:`LightingTimeline.evaluate_rgb01` does it -- ``clip(rgb + w, 0, 255)``, where the clip is
    load-bearing because a near-white cue overflows without it. The truncation to integers matches
    ``DroneSwarm._apply_drone_color``, which packs each channel with ``int()``, so quantization can
    only ever darken relative to intent.

    **Note: that truncation requirement is currently vacuous, and byte-level agreement holds for a
    different reason than it claims.** :meth:`LightingTimeline.evaluate` already rounds, so every
    value arriving here is integral and ``.astype(int)`` is bit-identical to
    ``np.round(...).astype(int)`` -- there is no fractional part left to truncate, and no test can
    tell the two apart. The browser and the hardware agree because both are handed the same already
    rounded WRGB, not because this line truncates. Keep the truncation anyway: it is what stays
    correct if ``evaluate`` ever stops rounding.

    Args:
        cues: One drone-deck's ``{time: (4,) WRGB}`` cue dict, as ``compile_cues`` returns it.

    Returns:
        ``{"times": [...], "rgb": [[r, g, b], ...]}``, JSON-serializable and index-parallel.
    """
    wrgb = np.stack(list(cues.values()))
    rgb = np.clip(wrgb[:, 1:] + wrgb[:, :1], 0, 255).astype(int)
    return {"times": list(cues), "rgb": rgb.tolist()}


class AppBackend:
    """Backend for choreography generation, filtering, preset storage, and deployment."""

    def __init__(
        self,
        *,
        music_dir: Path = Path(__file__).parents[2] / "music" / "songs",
        preset_dir: Path | None = None,
        config_file: Path | None = None,
        strict_processing: bool = True,
        strict_drone_match: bool = True,
        model_id: str = "gpt-5.6-luna",
        use_motion_primitives: bool = True,
        llm_provider: LLMProvider = "openai",
    ):
        """Initialize the backend by loading the music files and initializing the choreographer.

        Args:
            config_file: Path to the config file.
            music_dir: Path to the music directory.
            preset_dir: Path to the preset directory.
            strict_processing: Flag to raise an error on waypoint collisions.
            strict_drone_match: Flag to raise an error when preset drones do not match the current
                swarm.
            model_id: The OpenAI or Ollama model name (see LLM selector in the UI).
            use_motion_primitives: If we want LLM to use motion primitives for choreography
            llm_provider: ``openai`` or ``ollama`` for the choreographer backend.
        """
        self.root_path = Path(__file__).resolve().parents[2]
        self.preset_dir = preset_dir or self.root_path / "swarm_gpt/data/presets"
        with open(self.root_path / "swarm_gpt/data/settings.yaml", "r") as f:
            self.settings = yaml.safe_load(f)
        # Initialize drone control elements
        self.waypoints: Array | None = None  # High-level LLM commands
        self.splines = {}  # Low-level optimized commands from axswarm
        self.drone_controller = None  # TODO Controller for the Crazyflie drones
        # Initialize chat elements
        self.choreographer = Choreographer(
            config_file=config_file,
            model_id=model_id,
            llm_provider=llm_provider,
            use_motion_primitives=use_motion_primitives,
        )
        self.music_manager = MusicManager(music_dir)
        self.mode: Literal["preset", "real"] = "real"
        self._preset: None | str = None
        self._strict_processing = strict_processing
        self._strict_drone_match = strict_drone_match
        self._active_swarm: DroneSwarm | None = None
        if set(self.songs) & set(self.presets):
            raise ValueError("Songs and presets must have unique names")

    @property
    def songs(self) -> list[str]:
        """List of available songs."""
        return self.music_manager.songs

    @property
    def presets(self) -> list[str]:
        """List of available presets."""
        if not self.preset_dir.is_dir():
            return []
        return sorted(s.name for s in self.preset_dir.iterdir() if s.is_dir())

    @staticmethod
    def parse_preset_id(preset_id: str) -> dict[str, Any]:
        """Parse the preset directory name into display metadata."""
        try:
            song, n_drones, timestamp = [part.strip() for part in preset_id.rsplit("|", 2)]
            n_drones_int = int(n_drones)
        except ValueError:
            return {
                "id": preset_id,
                "song": preset_id,
                "numDrones": None,
                "createdAt": None,
                "createdLabel": None,
            }

        created_at = None
        created_label = timestamp
        try:
            created = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            created_at = created.isoformat()
            created_label = created.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            ...
        return {
            "id": preset_id,
            "song": song,
            "numDrones": n_drones_int,
            "createdAt": created_at,
            "createdLabel": created_label,
        }

    def preset_metadata(self, preset_id: str) -> dict[str, Any]:
        """Return display metadata for a preset id."""
        if preset_id not in self.presets:
            raise FileNotFoundError(f"Preset not found: {preset_id}")
        return self.parse_preset_id(preset_id)

    @self_correct(n_retries=2)
    def initial_prompt(self, song: str, *, response: str | None = None) -> list[dict[str, str]]:
        """Set the song and generate the choreography.

        Args:
            song: Name of the song or preset to use.
            response: Optional, predefined response. Used for testing.

        Returns:
            The chat history as a list of dictionaries with the role and content.
        """
        logger.info(f"Generating initial choreography for song: {song}")
        song_name = self._load_song(song)
        structure = self._load_structure(song_name)
        self.choreographer.reset_history()
        prompt = self.choreographer.format_initial_prompt(song_name, structure)

        fixed_response = response is not None
        if preset := song in self.presets:  # Preset was provided
            logger.debug(f"Loading preset: {song}")
            response = self.load_preset(song)
        elif fixed_response:  # Response was provided, do not use LLM
            logger.debug(f"Using predefined response: {response}")
            self.choreographer.messages.append({"role": "assistant", "content": response})
        else:  # Use LLM to generate the choreography
            logger.debug(f"Using LLM to generate choreography for song: {song_name}")
            response = self.choreographer.generate_choreography(prompt, structure=structure)

        try:
            self.waypoints = self.choreographer.response2waypoints(
                response, structure, strict=self._strict_processing
            )
            # The lighting cannot be compiled yet -- a look freezes a position snapshot, and
            # positions exist only after the axswarm pass. Checking the half that needs no
            # positions here is what lets a malformed lighting track reprompt.
            self.choreographer.validate_lighting(response)
        except LLMException as e:
            # We do not want to retry if we are using a preset or a fixed response. This
            # would use the LLM. We raise an error type that is not caught by
            # self_correct to exit immediately.
            if preset or fixed_response:
                raise RuntimeError("Initial prompt failed") from e
            raise e
        logger.info("Successfully generated choreography")
        return self.choreographer.messages

    @self_correct(n_retries=3)
    def reprompt(self, message: str) -> list[dict[str, str]]:
        """Reprompt the LLM to generate new waypoints based on the previous choreography.

        Args:
            message: The reprompt.

        Returns:
            The chat history as a list of dictionaries with the role and content.
        """
        logger.info(f"Reprompting with message: {message}")
        if message == "":
            logger.warning("No message provided, returning current history")
            return self.choreographer.messages
        prompt = self.choreographer.format_reprompt(message)
        structure = self._load_structure(self.music_manager.song)
        response = self.choreographer.generate_choreography(prompt, structure=structure)
        self.waypoints = self.choreographer.response2waypoints(
            response, structure, strict=self._strict_processing
        )
        self.choreographer.validate_lighting(response)
        logger.info("Successfully generated choreography")
        return self.choreographer.messages

    def simulate(self, gui: bool = False) -> dict[str, Any]:
        """Run the simulation with waypoints generated by the choreographer.

        Before the simulation is run, the waypoints are interpolated by axswarm to ensure that the
        trajectories are collision-free.

        Args:
            gui: Whether to show the MuJoCo debug replay after filtering. Use for debugging only.

        Returns:
            A collection of data from the simulation.
        """
        logger.info("Simulating trajectories with axswarm")
        assert self.waypoints is not None, "Please generate a choreography first"

        for key, data, total in simulate_axswarm(self.waypoints, self.settings, gui=False):
            if key == "progress":
                yield key, data, total
            else:
                sim_data = data
                break
        t = sim_data["timestamps"][::5]  # TODO remove hard coded downsampling factor
        lam = 0.1  # TODO: Adjust the smoothing parameters
        self.splines.clear()
        for i, drone in self.choreographer.agents.items():
            controls = sim_data["controls"][:, i, :3]
            self.splines[drone] = make_smoothing_spline(t, controls, lam=lam)
        if gui:
            replay_sim_states(sim_data, self.settings, self.lighting_timeline(), self.music_manager)
        logger.info("Simulation successful")
        return sim_data

    def lighting_timeline(self) -> LightingTimeline:
        """Compile the current response's lighting track into a timeline.

        Each look freezes a position snapshot taken from the axswarm splines, so this is only
        available once :meth:`simulate` has run.

        Every read-out goes through here, including for a response carrying no lighting at all:
        that compiles to the default hue wheel, so the preview and the hardware still agree.

        Returns:
            The compiled timeline, covering the whole flight.
        """
        assert self.splines, "Please run the simulation first!"
        assert self.waypoints is not None, "Please generate a choreography first"
        structure = self._load_structure(self.music_manager.song)

        def position_at(t: float) -> Array:
            return np.array([self.splines[i](t) for i in sorted(self.splines)])

        return self.choreographer.response2lighting(
            self._response_text(), structure, position_at, float(self.waypoints["time"][0, -1])
        )

    def browser_cues(self) -> dict[str, list[dict[str, list]]]:
        """Adapt the compiled lighting cues for the browser viewer.

        The browser is the third read-out, and it plays back the same baked cue list the hardware
        does, so the preview shows the ``col_freq`` quantization that will actually fly. That is
        deliberately *not* the sim's per-frame ``evaluate``, which is smoother than either.

        ``compile_cues`` output is not browser-ready: it is URI-keyed, holds ``NDArray`` values in
        ``{time: wrgb}`` dicts, and carries four channels. This is the whole of the adaptation, so
        ``normalize_playback`` stays a pure reshaper and deploy's ``col_freq``/``t_end`` plumbing
        does not spread into the API layer.

        **Known divergence: editing `lighting.toml` between preview and flight desynchronizes
        them.** This runs once per job and caches into ``job.playback``, while :meth:`deploy`
        recompiles from scratch and ``load_lighting_config`` re-reads the file every call with no
        cache, so the viewer can show the old palette while the drones fly the new one. Re-running
        the job resyncs. Nothing else can diverge: there is no RNG in the lighting path, and both
        paths take ``t_end``, ``col_freq``, the response text and the positions from one source.

        Returns:
            ``{"top": [...], "bot": [...]}``, each a list of one ``{"times", "rgb"}`` entry per
            drone, indexed like the payload's ``states`` rows rather than keyed by radio URI.
        """
        cfg = load_lighting_config()
        # Index keys, not radio URIs: `compile_cues` keys its output by whatever it is handed, and
        # a browser payload has no business carrying radio addresses.
        keys = [str(i) for i in range(self.choreographer.num_drones)]
        decks = compile_cues(
            self.lighting_timeline(), keys, cfg.col_freq, float(self.waypoints["time"][0, -1])
        )
        return {
            deck: [_fold_cues_to_rgb(cues[key]) for key in keys]
            for deck, cues in zip(("top", "bot"), decks, strict=True)
        }

    def _response_text(self) -> str:
        """The assistant response the current waypoints and splines were generated from."""
        assert self.choreographer.messages, "Please generate a choreography first"
        message = self.choreographer.messages[-1]
        assert message["role"] == "assistant", "Last message in history is not a response"
        return message["content"]

    def deploy(self, drone_ids: list[int] | None = None) -> bool:
        """Run the Crazyflie drones with waypoints generated by the choreographer.

        We call the waypoint_helpers.py script from the Crazyflie ROS package to run the drones.

        Returns:
            The chat history as a list of prompts and answers.
        """
        # Check if even in deploy environment
        if not self.settings["lighthouse"]:
            try:
                import rclpy

                if not rclpy.ok():
                    rclpy.init()  # Do it only once to be able to deploy multiple times
            except ImportError as _:
                logger.error("ROS2 is not installed. Switch to deploy environment!")
                return False

        from swarm_gpt.core.drone_swarm import DroneSwarm

        logger.info("Deploying drones")
        assert self.splines, "Please run the simulation first!"

        play_start_s, play_end_s = self.crop_window(self.music_manager.song)

        if not self.music_manager.verify_libvlc():
            logger.error("VLC/libvlc is not available. Install VLC (see README) before deploying.")
            return False

        # Bake the lighting before connecting any radio, so a malformed track fails cheaply.
        # `cfg.col_freq` reaches both the cue consumer and the cue compiler from the same config
        # field, so the Nyquist clamp in `build_look` can never disagree with the rate the cues
        # are actually drained at. A response with no lighting track compiles from the default
        # hue wheel, which is the one-colour-then-black cue pair the deploy path always sent.
        cfg = load_lighting_config()
        t_end = float(self.waypoints["time"][0, -1])
        uris = [d["uri"] for d in self.choreographer.drones.values()]
        color_top, color_bot = compile_cues(self.lighting_timeline(), uris, cfg.col_freq, t_end)

        swarm = DroneSwarm(
            self.choreographer.drones, col_freq=cfg.col_freq, lighthouse=self.settings["lighthouse"]
        )
        self._active_swarm = swarm
        logger.info("Swarm connected...")

        # generate references
        correct_positions = True
        init_pos_dict = {}
        final_pos_dict = {}
        landing_pos_dict = {}
        choreography_dict = {}
        for i, d in enumerate(self.choreographer.drones.values()):
            uri = d["uri"]
            init_pos = np.array(self.splines[i](0))
            obs = swarm.get_obs(uri)
            if np.linalg.norm(obs["pos"] - d["pos"]) > 0.3:
                correct_positions = False
                logger.warning(
                    f"Drone {uri} is too far from the expected initial position. pos={obs['pos']}, exp={d['pos']}"
                )
            landing_pos = obs["pos"] if self.settings["land_on_docks"] else d["pos"]
            # TODO fix hard coded yaw
            init_pos_dict[uri] = np.array([*init_pos, 0.0])
            final_pos_dict[uri] = np.array([*landing_pos + np.array([0.0, 0.0, 0.5]), 0.0])
            landing_pos_dict[uri] = np.array([*landing_pos - np.array([0.0, 0.0, 0.2]), 0.0])
            choreography_dict[uri] = self.splines[i]

        try:
            if not correct_positions:
                raise RuntimeError("Some drone(s) are not in the expected initial positions.")
            swarm.goto(init_pos_dict)  # takeoff
            # Check active drones after the initial climb.
            taken_off = True
            for d in self.choreographer.drones.values():
                uri = d["uri"]
                if not swarm.is_active(uri):
                    logger.warning(f"Drone {uri} is inactive after takeoff")
                    taken_off = False
                    continue
                try:
                    logger.debug(f"Trying to get obs for {uri}")
                    obs = swarm.get_obs(uri)
                    logger.debug(f"got obs for {uri}")
                    z = obs["pos"][2]
                    qw = np.abs(obs["quat"][-1])
                # Demo fix: If the drone is disconnected, we cannot get its position. We assume it has not taken off.
                # TODO: Replace the general exception catch with the specific cflib2 exception.
                except Exception as e:
                    logger.warning(f"Could not get position for drone {uri} after takeoff: {e}")
                    taken_off = False
                    continue
                if z < 0.2 or qw < 0.8:
                    taken_off = False
                    logger.warning(f"Drone {uri} has not taken off yet: z={z:.2f}m, qw={qw:.2f}")
            if taken_off:
                if not self.music_manager.play(wait=True, start_s=play_start_s, end_s=play_end_s):
                    logger.error(
                        "VLC could not start playback; skipping choreography (drones will land)."
                    )
                else:
                    logger.debug("Starting choreography execution")
                    swarm.execute_choreography(
                        choreography_dict,
                        self.waypoints["time"][0, -1],
                        color_top=color_top,
                        color_bot=color_bot,
                    )
            self.music_manager.stop()
            swarm.goto(final_pos_dict, duration=2.0)  # Transition from ideal point to hover pos
            if self.settings["land_on_docks"]:
                swarm.goto(final_pos_dict, duration=3.0)  # Hovering
            swarm.land(duration=1.5)  # Landing
        finally:
            self._active_swarm = None
            swarm.close()
        logger.info("Deployment successful")
        return True

    def emergency_stop_active_swarm(self) -> None:
        """Emergency-stop the currently active deployment swarm, if one exists."""
        swarm = self._active_swarm
        if swarm is None:
            raise RuntimeError("No active deployment swarm to emergency stop.")
        swarm.emergency_stop()
        self.music_manager.stop()

    def load_preset(self, preset_id: str) -> str:
        """Load a preset response.

        Args:
            preset_id: Name of the preset.
        """
        assert preset_id, "Please select a valid preset"
        assert preset_id in self.presets, "No preset for this song"
        preset_path = self.preset_dir / preset_id
        n_drones = self.choreographer.num_drones
        preset_n_drones = int(preset_id.rsplit("|", 2)[1].strip())
        if preset_n_drones != n_drones and self._strict_drone_match:
            raise ValueError(
                f"Preset n_drones ({preset_n_drones}) do not match current swarm ({n_drones})"
            )
        with open(preset_path / "history.json", "r") as f:
            history = json.load(f)
        with open(preset_path / "meta.json", "r") as f:
            meta = json.load(f)
        if meta["use_motion_primitives"] != self.choreographer.use_motion_primitives:
            raise ValueError("Preset was generated with a different use_motion_primitives setting")
        assert history[-1]["role"] == "assistant", "Last message in history is not a response"
        self.choreographer.messages = history
        return history[-1]["content"]

    def save_preset(self) -> str:
        """Save the preset."""
        if not self.choreographer.messages:
            raise ValueError("No preset to save. Run Simulation first")
        if self.waypoints is None or not self.splines:
            raise ValueError("No safe preset to save. Run the safety filter first")

        self.preset_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        for offset_seconds in range(100):
            timestamp = datetime.fromtimestamp(now.timestamp() + offset_seconds).strftime(
                "%Y%m%d_%H%M%S"
            )
            preset_name = (
                self.music_manager.song + f" | {self.choreographer.num_drones} | {timestamp}"
            )
            path = self.preset_dir / preset_name
            if not path.exists():
                break
        else:
            raise FileExistsError("Could not create a unique preset name")
        path.mkdir(parents=True)

        with open(path / "history.json", "w") as f:
            json.dump(self.choreographer.messages, f)
        meta = {"n_drones": self.choreographer.num_drones, "song": self.music_manager.song}
        meta["use_motion_primitives"] = self.choreographer.use_motion_primitives
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)
        if self.waypoints is not None:
            np.save(path / "waypoints.npy", self.waypoints)

        pos_splines = self.splines
        vel_splines = {i: s.derivative() for i, s in pos_splines.items()}
        acc_splines = {i: s.derivative() for i, s in vel_splines.items()}
        des_time = np.arange(0, self.waypoints["time"][0, -1], 1.0 / self.settings["state_freq"])
        des_pos = [s(des_time) for s in pos_splines.values()]
        des_vel = [s(des_time) for s in vel_splines.values()]
        des_acc = [s(des_time) for s in acc_splines.values()]
        des_pos = np.array(des_pos).swapaxes(0, 1)
        des_vel = np.array(des_vel).swapaxes(0, 1)
        des_acc = np.array(des_acc).swapaxes(0, 1)

        N = des_time.shape[0]
        M = self.choreographer.num_drones

        # Build combined array: time | pos (M*3) | vel (M*3)
        header = ["time[s]"]
        combined = np.zeros((N, 1 + 6 * M), dtype=float)
        combined[:, 0] = des_time

        for i in range(M):
            combined[:, 6 * i + 1 : 6 * i + 4] = des_pos[:, i, :]
            combined[:, 6 * i + 4 : 6 * i + 7] = des_vel[:, i, :]
            header += [f"drone{i}_posx[m]", f"drone{i}_posy[m]", f"drone{i}_posz[m]"]
            header += [f"drone{i}_velx[m/s]", f"drone{i}_vely[m/s]", f"drone{i}_velz[m/s]"]

        header_str = ",".join(header)

        csv_path = path / "trajectory.csv"
        np.savetxt(csv_path, combined, delimiter=",", header=header_str, comments="", fmt="%.6f")
        logger.info("Saved trajectory CSV: %s", csv_path)
        return preset_name

    def delete_preset(self, preset_id: str) -> None:
        """Delete a saved preset directory."""
        if preset_id not in self.presets:
            raise FileNotFoundError(f"Preset not found: {preset_id}")
        preset_root = self.preset_dir.resolve()
        preset_path = (self.preset_dir / preset_id).resolve()
        if not preset_path.is_dir() or not preset_path.is_relative_to(preset_root):
            raise FileNotFoundError(f"Preset not found: {preset_id}")
        shutil.rmtree(preset_path)

    def _load_song(self, song: str) -> str:
        """Load the song on the music manager."""
        if song in self.presets:
            song = self.parse_preset_id(song)["song"]
        self.music_manager.song = song
        return song

    def crop_window(self, song_name: str) -> tuple[float, float]:
        """Return the ``(start_s, end_s)`` crop window for a song, in seconds.

        Reads ``song_crops`` from settings, falling back to ``song_crops.default`` for any song
        without an explicit entry.

        Args:
            song_name: Stem of the MP3 file (no extension).

        Returns:
            The ``(start_s, end_s)`` window the song is cropped to.
        """
        crops = self.settings["song_crops"]
        window = crops.get(song_name, crops["default"])
        return float(window[0]), float(window[1])

    def _load_structure(self, song_name: str) -> SongStructure:
        """Load the cached SongStructure JSON for a song, cropped to its window.

        The full-song analysis is loaded from disk and then cropped to the song's
        ``song_crops`` window (see :meth:`crop_window`); only that window is choreographed.

        Args:
            song_name: Stem of the MP3 file (no extension).

        Raises:
            FileNotFoundError: If no analysis JSON exists yet for the song.
        """
        json_path = self.root_path / "music" / "analyzed" / f"{song_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"No analysis found for '{song_name}'. Run `pixi run -e music analyze` first."
            )
        start_s, end_s = self.crop_window(song_name)
        return SongStructure.from_json(json_path).crop(start_s, end_s)

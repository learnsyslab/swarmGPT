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
from swarm_gpt.core.sim import replay_sim_states, simulate_axswarm
from swarm_gpt.exception import LLMException
from swarm_gpt.utils import MusicManager, generate_default_colors

if TYPE_CHECKING:
    from numpy.typing import NDArray as Array

    from swarm_gpt.utils.llm_providers import LLMProvider

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

colors = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.7, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.5],
]

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


class AppBackend:
    """Backend for choreography generation, filtering, preset storage, and deployment."""

    def __init__(
        self,
        *,
        music_dir: Path = Path(__file__).parents[2] / "music",
        preset_dir: Path | None = None,
        config_file: Path | None = None,
        strict_processing: bool = True,
        strict_drone_match: bool = True,
        model_id: str = "gpt-4o",
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
        music_info = self.music_manager.extract_song_info()
        self.choreographer.reset_history()
        prompt = self.choreographer.format_initial_prompt(song_name, music_info)

        fixed_response = response is not None
        if preset := song in self.presets:  # Preset was provided
            logger.debug(f"Loading preset: {song}")
            response = self.load_preset(song)
        elif fixed_response:  # Response was provided, do not use LLM
            logger.debug(f"Using predefined response: {response}")
            self.choreographer.messages.append({"role": "assistant", "content": response})
        else:  # Use LLM to generate the choreography
            logger.debug(f"Using LLM to generate choreography for song: {song_name}")
            response = self.choreographer.generate_choreography(prompt)

        try:
            self.waypoints = self.choreographer.response2waypoints(
                response, music_info=music_info, strict=self._strict_processing
            )
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
        music_info = self.music_manager.extract_song_info()
        response = self.choreographer.generate_choreography(prompt)
        self.waypoints = self.choreographer.response2waypoints(
            response, music_info=music_info, strict=self._strict_processing
        )
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
            replay_sim_states(sim_data, self.settings, self.music_manager)
        logger.info("Simulation successful")
        return sim_data

    def deploy(self, drone_ids: list[int] | None = None) -> bool:
        """Run the Crazyflie drones with waypoints generated by the choreographer.

        We call the waypoint_helpers.py script from the Crazyflie ROS package to run the drones.

        Returns:
            The chat history as a list of prompts and answers.
        """
        # Check if even in deploy environment
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

        # If a deploy version of the song is present, play it
        original_song = self.music_manager.song
        try:
            self.music_manager.song = original_song + "[deploy]"
        except AssertionError:
            ...

        # generate references
        init_pos_dict = {}
        final_pos_dict = {}
        choreography_dict = {}
        colors_dict = {}
        colors_array = np.zeros((self.choreographer.num_drones, 4))
        colors_array[:, 1:] = generate_default_colors(self.choreographer.num_drones, limit=255)
        colors_array[:, 3] *= 0.8  # Dim blue channel since that LED is brighter

        for i, d in enumerate(self.choreographer.drones.values()):
            init_pos = np.array(self.splines[i](0))
            final_pos = d["pos"]  # + np.array([0.0, 0.0, 0.2])
            # TODO fix hard coded yaw
            init_pos_dict[d["uri"]] = [np.array([*init_pos, 0.0])]
            final_pos_dict[d["uri"]] = [np.array([*final_pos, 0.0])]
            choreography_dict[d["uri"]] = self.splines[i]
            colors_dict[d["uri"]] = {
                "t": np.array([0, 0.5, 1.0]),
                "color_top": np.array([colors_array[i], colors_array[i], colors_array[i]]),
                "color_bot": np.array([colors_array[i], colors_array[i], colors_array[i]]),
            }  # Default colors

        swarm = DroneSwarm(self.choreographer.drones, lighthouse=self.settings["lighthouse"])
        logger.info("Swarm connected...")
        try:
            # swarm.apply_colors(colors_dict)
            swarm.goto(init_pos_dict)
            # check if all drones have taken off
            taken_off = True
            for i, d in enumerate(self.choreographer.drones.values()):
                if not swarm.lighthouse and swarm.get_obs(d["uri"])["pos"][2] < 0.2:
                    taken_off = False
                    logger.warning(f"Drone {d['uri']} has not taken off yet")
            if taken_off:
                self.music_manager.play()
                swarm.execute_choreography(
                    choreography_dict, self.waypoints["time"][0, -1], colors_dict
                )
            swarm.goto(final_pos_dict, duration=3.0)
        finally:
            swarm.close()
        self.music_manager.song = original_song
        logger.info("Deployment successful")
        return True

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

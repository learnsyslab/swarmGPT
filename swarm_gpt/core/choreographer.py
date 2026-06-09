"""The choreographer module handles the interaction with the LLM."""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import einops  # pyright: ignore[reportMissingImports]
import numpy as np
import toml
import yaml

from swarm_gpt.core.motion_primitives import motion_primitives as motion_primitives_collection
from swarm_gpt.core.motion_primitives import primitive_by_name
from swarm_gpt.core.structured_output_schema import (
    KEY_PATTERN,
    build_motion_primitive_response_schema,
    decode_key,
    encode_key,
    structured_payload_to_choreography,
)
from swarm_gpt.exception import LLMFormatError, LLMPlanError, LLMResponseProcessingError
from swarm_gpt.utils.llm_providers import (
    RESPONSES_MAX_OUTPUT_TOKENS,
    RESPONSES_TEMPERATURE,
    cancellable_ollama_chat,
    openai_client_for_provider,
    prepare_responses_messages,
    register_ollama_client,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from openai import OpenAI

    from swarm_gpt.utils.llm_providers import LLMProvider
    from swarm_gpt.utils.music_analyzer import SongStructure

logger = logging.getLogger(__name__)

# Set to True to see raw LLM outputs in terminal
DEBUG_LLM_OUTPUT = True
# OLLAMA_CONTEXT_LENGTH = None  # Set None to use Ollama's VRAM-based default.
OLLAMA_CONTEXT_LENGTH = None


# Investigate and improve error message for the case when func = "", and we get key error, during sanitize llm output
# Also improve error message when there is an issue with function output, so that we can re-prompt with super specific messag
# Need to imorove parsing
# Add a log everytime some waypoint is clamped.
class Choreographer:
    """The choreographer handles the interaction with the language model.

    It formats the prompts and parses the output of the language model into the desired format.
    """

    def __init__(
        self,
        *,
        config_file: Path | None = None,
        model_id: str = "gpt-4o",
        llm_provider: LLMProvider = "openai",
        use_motion_primitives: bool = False,
    ):
        """Initialize the choreographer.

        Args:
            config_file: Path to the drone configuration file that is used for crazyswarm.
            model_id: Model name passed to ``responses.create`` (OpenAI id or Ollama tag).
            llm_provider: ``openai`` for the cloud API, ``ollama`` for local via OpenAI-compatible URL.
            use_motion_primitives: Whether to use motion primitives or raw waypoints.
        """
        self.settings = None
        self.llm_provider: LLMProvider = llm_provider
        self._model_id = model_id
        self._chat_client: OpenAI | None = None
        self._chat_client_provider: LLMProvider | None = None
        self.use_motion_primitives = use_motion_primitives
        self.agents = {}
        self.uris = {}
        self.starting_pos = {}
        self.num_drones = 0
        self.messages = []
        # Load prompts from file
        prompt = "motion_primitive_prompts" if self.use_motion_primitives else "prompts"
        with open(Path(__file__).resolve().parents[1] / f"data/{prompt}.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)
        self.load_drone_config(config_file)
        # Limits define boundaries of permissible flying area
        self.lim_lower = np.array(self.settings["axswarm"]["pos_min"])
        self.lim_upper = np.array(self.settings["axswarm"]["pos_max"])
        assert len(self.lim_lower) == 3 and len(self.lim_upper) == 3, "Limits must be 3D"
        # Stride (in bars) between required downbeats; beats in between are optional accents.
        self._bars_per_required = int(
            self.settings.get("choreography", {}).get("bars_per_required", 1)
        )

    def configure_llm(self, provider: LLMProvider, model_id: str) -> None:
        """Switch provider and model (used by the web UI).

        Keeps choreography history untouched; callers should reset when switching songs as today.
        """
        if provider != self.llm_provider:
            self._chat_client = None
            self._chat_client_provider = None
        self.llm_provider = provider
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        """Currently selected inference model id (OpenAI or Ollama tag)."""
        return self._model_id

    def _chat_client_for_call(self) -> OpenAI:
        if self._chat_client is None or self._chat_client_provider != self.llm_provider:
            try:
                self._chat_client = openai_client_for_provider(self.llm_provider)
            except RuntimeError as e:
                raise LLMPlanError(str(e)) from e
            self._chat_client_provider = self.llm_provider
            if self.llm_provider == "ollama":
                register_ollama_client(self._chat_client)
        return self._chat_client

    def format_initial_prompt(self, song: str, structure: SongStructure) -> list[dict[str, str]]:
        """Format the initial prompt for the LLM.

        Args:
            song: The name of the song.
            structure: Hierarchical music structure (segments / bars / beats).

        Returns:
            The formatted initial prompt as a list of role/content message dicts.
        """
        logger.debug("Formatting initial prompt")
        msgs = []
        user_prompt = self._format_initial_user_prompt(song, structure)
        msgs.append({"role": "system", "content": self.prompts["system_initial"]})
        msgs.append({"role": "user", "content": user_prompt})
        msgs.append({"role": "system", "content": self.prompts["example"]})
        output_format_key = (
            "output_format_structured"
            if self._uses_structured_outputs() and "output_format_structured" in self.prompts
            else "output_format"
        )
        msgs.append({"role": "system", "content": self.prompts[output_format_key]})
        return msgs

    def format_reprompt(self, message: str) -> list[dict[str, str]]:
        """Format the reprompt for the LLM."""
        logger.debug("Formatting reprompt")
        msgs = []
        msgs.append({"role": "user", "content": message})
        output_format_key = (
            "output_format_structured"
            if self._uses_structured_outputs() and "output_format_structured" in self.prompts
            else "output_format"
        )
        msgs.append({"role": "system", "content": self.prompts[output_format_key]})
        return msgs

    def _uses_structured_outputs(self) -> bool:
        """Use Structured Outputs for all motion-primitive providers."""
        return self.use_motion_primitives

    def generate_choreography(
        self, prompt: list[dict[str, str]], structure: SongStructure | None = None
    ) -> str:
        """Generate the initial choreography for the LLM.

        Args:
            prompt: The message list returned by :meth:`format_initial_prompt`.
            structure: Hierarchical song structure; required when using structured outputs.

        Returns:
            The assistant's response text (YAML-shaped for legacy parsing).
        """
        logger.debug(
            "Generating choreography with provider=%s model=%s", self.llm_provider, self._model_id
        )
        self.messages.extend(prompt)
        if self._uses_structured_outputs():
            if structure is None:
                raise ValueError("structure is required for structured output generation")
            payload = self._call_responses_structured(self.messages, structure=structure)
            response = self._structured_payload_to_text(payload)
        else:
            response = self._call_responses(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def reset_history(self):
        """Reset the LLM history to ensure a clean slate."""
        self.messages.clear()

    def load_drone_config(self, config_file: Path | None = None):
        """Load the drone configuration from the config file.

        The configuration file is a yaml file that contains the drone IDs and their initial
        positions.
        """
        with open(Path(__file__).resolve().parents[1] / "data/settings.yaml", "r") as f:
            self.settings = yaml.safe_load(f)

        if config_file is None:
            config_file = Path(__file__).resolve().parents[1] / "data/drones.toml"
        with open(config_file) as f:
            self.drones = toml.load(f)

        for drone_name, data in self.drones.items():
            i = int(drone_name[2:])
            self.agents[i] = i
            self.starting_pos[i] = np.array(data["pos"])
            self.starting_pos[i][2] = self.settings["starting_height"]
            self.uris[i] = data["uri"]
        self.num_drones = len(self.agents.values())
        assert self.num_drones > 0, "No drones detected in config file"

    def _format_initial_user_prompt(self, song: str, structure: SongStructure) -> str:
        """Format the initial user prompt for the LLM.

        Args:
            song: The name of the song.
            structure: Hierarchical song structure used to render segments/keys.
        """
        # Convert starting positions to cm for the LLM (integer tokens).
        starting_pos = [(pos * 100).astype(int).tolist() for pos in self.starting_pos.values()]
        segments_table = _render_segments_table(structure)
        required_keys_csv = ", ".join(
            encode_key(*k) for k in structure.required_keys(self._bars_per_required)
        )
        n_total_beats = len(structure.all_keys())
        if self.use_motion_primitives:
            latex_file = Path(__file__).resolve().parents[1] / "data/latex_eqn.yaml"
            with open(latex_file, "r") as file:
                data = yaml.safe_load(file)
        prompt_kwargs = {
            "song": song,
            "bpm": structure.bpm,
            "num_drones": self.num_drones,
            "starting_pos": starting_pos,
            "segments_table": segments_table,
            "required_keys_csv": required_keys_csv,
            "n_total_beats": n_total_beats,
            "lim_lower": (self.lim_lower * 100).astype(int).tolist(),
            "lim_upper": (self.lim_upper * 100).astype(int).tolist(),
            "move_z_typical_cm": int((self.lim_upper[2] - self.lim_lower[2]) * 100 / 2),
            "wave_eqn": data["wave"] if self.use_motion_primitives else None,
        }
        return self.prompts["user_initial"].format(**prompt_kwargs)

    def _call_responses(self, messages: list[dict[str, str]]) -> str:
        """Call ``responses.create`` (OpenAI cloud or Ollama's OpenAI-compatible endpoint)."""
        client = self._chat_client_for_call()
        input_messages, instructions = prepare_responses_messages(messages)
        try:
            response = client.responses.create(
                model=self._model_id,
                input=input_messages,
                instructions=instructions,
                max_output_tokens=RESPONSES_MAX_OUTPUT_TOKENS,
                temperature=RESPONSES_TEMPERATURE,
            )
        except Exception as e:
            hint = (
                "Ensure `ollama serve` is running and the model is pulled."
                if self.llm_provider == "ollama"
                else "Check OPENAI_API_KEY and model availability."
            )
            raise LLMPlanError(
                f"Responses API call failed for provider={self.llm_provider!r} "
                f"model={self._model_id!r}. {hint} ({e})"
            ) from e
        if response.error is not None:
            raise LLMPlanError(
                f"Model {self._model_id!r} returned an error: {response.error.message}"
            )
        content = response.output_text
        if not content:
            raise LLMPlanError(
                f"Model {self._model_id!r} returned empty content. Try another model or reprompt."
            )
        if DEBUG_LLM_OUTPUT:
            print("\n" + "=" * 80)
            print("RAW LLM OUTPUT:")
            print(content)
            print("=" * 80 + "\n")
        return content

    def _collision_check(self, pos: NDArray, min_dist: float = 0.1):
        """Check that no two drones are too close to each other at the same time.

        Args:
            pos: The positions of the drones as a (n_drones, T, 3) array.
            min_dist: The minimum allowed distance between any two drones at the same time.

        Raises:
            ValueError: If two drones are too close together at the same time.
        """
        differences = pos[:, None, :, :] - pos[None, :, :, :]  # Reshape for broadcasting
        distance = np.linalg.norm(differences, axis=-1)
        # Set the diagonal to a large number to avoid comparing the same drone
        distance += np.eye(self.num_drones).reshape(self.num_drones, self.num_drones, 1) * 1000
        min_distance = np.min(distance, axis=1)  # (n_drones, T). Closest encounter for each time
        if np.any(min_distance < min_dist):
            drones, times = np.nonzero(min_distance < min_dist)
            drones, times = drones.tolist(), times.tolist()
            raise LLMPlanError(f"Drones {set(drones)} get too close at waypoints {set(times)}")

    def _call_responses_structured(
        self, messages: list[dict[str, str]], structure: SongStructure
    ) -> dict:
        """Call Responses API with strict json_schema and parse JSON output."""
        if self.llm_provider == "ollama":
            return self._call_ollama_structured(messages, structure)

        client = self._chat_client_for_call()
        schema = build_motion_primitive_response_schema(
            all_keys=structure.all_keys(),
            required_keys=structure.required_keys(self._bars_per_required),
            num_drones=self.num_drones,
        )
        input_messages, instructions = prepare_responses_messages(messages)
        try:
            response = client.responses.create(
                model=self._model_id,
                input=input_messages,
                instructions=instructions,
                max_output_tokens=RESPONSES_MAX_OUTPUT_TOKENS,
                temperature=RESPONSES_TEMPERATURE,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "swarmgpt_choreography",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        except Exception as e:
            hint = (
                "Ensure `ollama serve` is running and the model supports json_schema response format."
                if self.llm_provider == "ollama"
                else "Check OPENAI_API_KEY and model availability."
            )
            raise LLMPlanError(
                f"Structured output call failed for provider={self.llm_provider!r} "
                f"model={self._model_id!r}. {hint} ({e})"
            ) from e
        if response.error is not None:
            raise LLMPlanError(
                f"Model {self._model_id!r} returned an error: {response.error.message}"
            )
        content = response.output_text
        if not content:
            raise LLMPlanError(f"Model {self._model_id!r} returned empty structured content.")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMFormatError(f"Structured output was not valid JSON: {e}") from e

    def _call_ollama_structured(
        self, messages: list[dict[str, str]], structure: SongStructure
    ) -> dict:
        """Call Ollama's native chat structured output path."""
        schema = build_motion_primitive_response_schema(
            all_keys=structure.all_keys(),
            required_keys=structure.required_keys(self._bars_per_required),
            num_drones=self.num_drones,
        )
        grounded_messages = [
            *messages,
            {
                "role": "system",
                "content": (
                    "Return valid JSON only. Match the provided response format exactly. "
                    "Use named params objects, never positional args arrays."
                ),
            },
        ]
        try:
            response = cancellable_ollama_chat(
                model=self._model_id,
                messages=grounded_messages,
                format=schema,
                options={"temperature": RESPONSES_TEMPERATURE, "num_ctx": OLLAMA_CONTEXT_LENGTH}
                if OLLAMA_CONTEXT_LENGTH is not None
                else {"temperature": RESPONSES_TEMPERATURE},
            )
        except Exception as e:
            raise LLMPlanError(
                f"Ollama structured chat failed for model={self._model_id!r}. "
                "Ensure `ollama serve` is running and the model is pulled. "
                f"({e})"
            ) from e
        content = self._extract_ollama_content(response)
        if not content:
            raise LLMPlanError(f"Model {self._model_id!r} returned empty structured content.")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMFormatError(f"Structured output was not valid JSON: {e}") from e

    @staticmethod
    def _extract_ollama_content(response: object) -> str:
        """Extract message text from Ollama responses in dict or object form."""
        if isinstance(response, dict):
            message = response.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, (dict, list)):
                    return json.dumps(content)
            content = response.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, (dict, list)):
                return json.dumps(content)
            return ""
        message = getattr(response, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            return json.dumps(content)
        return ""

    def _structured_payload_to_choreography(self, payload: dict) -> dict[tuple[int, int, int], str]:
        """Convert structured payload to a (seq, bar, beat) -> action-string dict."""
        return structured_payload_to_choreography(payload)

    def _structured_payload_to_text(self, payload: dict) -> str:
        """Convert structured payload to legacy YAML-like text for downstream parsing/history."""
        required_fields = ["song_mood", "choreography_plan", "choreography"]
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise LLMFormatError(
                "Structured output is missing required keys: " + ", ".join(sorted(missing))
            )
        choreography = self._structured_payload_to_choreography(payload)
        lines = [
            f"song_mood: {json.dumps(payload['song_mood'])}",
            f"choreography_plan: {json.dumps(payload['choreography_plan'])}",
            "choreography:",
        ]
        # Sort by (seq, bar, beat) tuple for deterministic, time-ordered output.
        for addr in sorted(choreography):
            lines.append(f"  {encode_key(*addr)}: {choreography[addr]}")
        lines.append("  END")
        return "\n".join(lines)

    def response2waypoints(
        self, text: str, structure: SongStructure, strict: bool = True, t_rth: float = 3.0
    ) -> dict[str, NDArray]:
        """Translate the LLM output into waypoints.

        Args:
            text: The output of the LLM, in the YAML-like form produced by the prompt.
            structure: Hierarchical song structure used to resolve action keys to seconds.
            strict: Enable/disable waypoint proximity and distance checks.
            t_rth: Time for the drones to return to their starting position.

        Returns:
            The waypoints as a dictionary of "time", "pos", "vel", "acc". "time" has shape
            (n_drones, T), and "pos", "vel", "acc" have shape (n_drones, T, 3).
        """
        logger.debug("Converting LLM output into waypoints")
        if self.use_motion_primitives:
            choreo = self._response2choreo(text, structure)
            waypoints = self._choreo2waypoints(choreo, structure)
        else:
            # Raw-waypoint mode predates the structure rewrite and still expects a flat
            # beat_times list. Pull it from the structure for the time being.
            flat_times = [b.time_s for s in structure.segments for bar in s.bars for b in bar.beats]
            waypoints = self._raw_response2waypoints(text, np.asarray(flat_times), structure)
        # Clip waypoint values to the physical limits
        waypoints["pos"] = np.clip(waypoints["pos"], self.lim_lower, self.lim_upper)
        if strict:
            self._collision_check(waypoints["pos"], 0.25)

        # Add home position (TODO make cleaner)
        home = np.zeros((len(self.agents.values()), 1, 3))
        for i in self.agents.values():
            home[i, :, :] = self.starting_pos[i]
        waypoints["time"] = np.concat(
            (
                waypoints["time"],
                waypoints["time"][:, -1][:, None] + t_rth,
                waypoints["time"][:, -1][:, None] + t_rth + 1.0,
            ),
            axis=1,
        )
        waypoints["pos"] = np.concat((waypoints["pos"], home, home), axis=1)

        return waypoints

    def _response2choreo(
        self, text: str, structure: SongStructure | None = None
    ) -> dict[tuple[int, int, int], str]:
        """Translate the LLM output into a (seq, bar, beat) -> action-string choreography."""
        assert self.use_motion_primitives, "Motion primitives not set in _response2choreo"
        choreography = self._slice_choreography_from_text(text, structure)
        # PLAN is not in the new schema but tolerated for legacy / preset payloads.
        for addr, moves in list(choreography.items()):
            if any(k in moves for k in ["helix", "spiral", "zig_zag", "wave"]) and moves.endswith(
                "PLAN"
            ):
                moves = moves.replace("PLAN", "").strip()
                moves = moves.replace("-", "").strip()
            elif moves.count("PLAN") > 1:
                moves = "PLAN"
            choreography[addr] = moves
        return choreography

    def _choreo2waypoints(
        self, choreography: dict[tuple[int, int, int], str], structure: SongStructure
    ) -> dict[str, np.ndarray]:
        """Translate a (seq, bar, beat)-keyed choreography into time-based waypoints.

        Resolves each hierarchical key to its absolute time via :meth:`SongStructure.time_of`,
        sorts actions by time, and renumbers them 1..N as synthetic indices for the existing
        time-based primitive execution pipeline.

        Args:
            choreography: Action strings keyed by ``(seq, bar, beat)``.
            structure: Song structure providing ``time_of`` and ``required_keys``.

        Returns:
            Waypoints dict with ``time``, ``pos``, ``vel``, ``acc`` arrays.
        """
        required = set(structure.required_keys(self._bars_per_required))
        emitted = set(choreography)
        if missing := required - emitted:
            raise LLMResponseProcessingError(
                f"Choreography is missing required keys at {sorted(missing)}"
            )
        if not choreography:
            raise LLMResponseProcessingError("Choreography is empty")

        # Sort emitted actions by their resolved time.
        ordered = sorted(
            (
                (structure.time_of(seq, bar, beat), (seq, bar, beat), action_str)
                for (seq, bar, beat), action_str in choreography.items()
            ),
            key=lambda triple: triple[0],
        )

        motion_primitives: dict[int, list[dict[str, tuple]]] = {}
        for synth_idx, (_time, addr, action_str) in enumerate(ordered, start=1):
            motion_primitives[synth_idx] = []
            moves = action_str.strip(" ;").split(";")
            for move in moves:
                fn_name = move.split("(")[0].strip(" -\n")
                if fn_name == "PLAN":
                    motion_primitives[synth_idx].append({fn_name: ()})
                    continue
                if fn_name not in motion_primitives_collection:
                    raise LLMResponseProcessingError(
                        f"Unknown motion primitive '{fn_name}' at {encode_key(*addr)}"
                    )
                # Parse `args` portion: ast.literal_eval on a re-wrapped tuple expression. The
                # trailing comma forces single-arg functions to parse as length-1 tuples.
                try:
                    fn_args = ast.literal_eval("(" + move.split("(")[1].split("#")[0][:-1] + ",)")
                except (SyntaxError, ValueError) as e:
                    raise LLMFormatError(
                        f"Cannot interpret arguments of '{move}' at {encode_key(*addr)}. "
                        f"Failed with {e.__class__.__name__}: {e}"
                    )
                n_args = motion_primitives_collection[fn_name.lower()]["n_args"]
                if len(fn_args) != n_args:
                    raise LLMFormatError(
                        f"{fn_name} at {encode_key(*addr)} must have {n_args} arguments, "
                        f"got {fn_args}"
                    )
                motion_primitives[synth_idx].append({fn_name: fn_args})

        timestamps = np.array([t for t, _, _ in ordered])
        # Forward-looking semantics: the last emitted primitive plays until the song ends.
        t_end = structure.segments[-1].end_s
        if t_end <= timestamps[-1]:
            t_end = float(timestamps[-1]) + 1.0  # guard against a zero-length final interval
        t, pos = self._motion_primitives2time_and_pos(motion_primitives, timestamps, t_end)
        return {"time": t, "pos": pos, "vel": np.zeros_like(pos), "acc": np.zeros_like(pos)}

    def _raw_response2waypoints(
        self, text: str, timestamps: NDArray, structure: SongStructure | None = None
    ) -> dict[int, np.ndarray]:
        """Translate the raw LLM output into waypoints."""
        assert not self.use_motion_primitives, "Motion primitives set in raw response processing"
        choreography = self._slice_choreography_from_text(text, structure)
        if missing := set(range(1, len(timestamps) + 1)) - set(choreography.keys()):
            raise LLMResponseProcessingError(f"Choreography plan is missing waypoints {missing}")

        for i, positions in choreography.items():
            try:
                # literal_eval is safe because it only supports a restricted subset of python
                positions = ast.literal_eval(positions)
            except (SyntaxError, ValueError):
                raise LLMFormatError(f"Cannot interpret waypoint {i} as a list (got {positions})")
            if not all(len(pos) == 3 for pos in positions):
                raise LLMResponseProcessingError("Waypoints must have 3 columns for x, y, z")

        positions = np.array([ast.literal_eval(p) for p in choreography.values()], dtype=np.float64)
        positions /= 100.0  # Convert back to meters. TODO: Remove all conversions
        positions = einops.rearrange(positions, "t d c -> d t c")
        start_pos = np.array(list(self.starting_pos.values()))
        pos = np.concatenate((start_pos[:, None, :], positions), axis=1)
        t = np.tile(np.concatenate(([0], timestamps)), (pos.shape[0], 1))
        return {"time": t, "pos": pos, "vel": np.zeros_like(pos), "acc": np.zeros_like(pos)}

    @staticmethod
    def _slice_choreography_from_text(
        text: str, structure: SongStructure | None = None
    ) -> dict[tuple[int, int, int], str]:
        """Extract the choreography from the YAML output of the LLM.

        The LLM output may not be valid YAML (formatting, quotes, dashes). We slice the
        ``choreography`` block manually and parse hierarchical keys of the form
        ``s<seq>b<bar>t<beat>``.

        Args:
            text: The YAML output of the LLM.
            structure: Optional song structure used to annotate each key with its resolved
                ``time_of`` value in the debug print. Has no effect on parsing.

        Returns:
            Dict mapping ``(seq, bar, beat)`` tuples to action strings.
        """
        yaml_text = re.findall(r"```yaml\n(.*?)(?:```)", text, re.DOTALL)
        try:
            yaml_text = yaml_text[0]
        except IndexError:
            yaml_text = text

        if DEBUG_LLM_OUTPUT:
            debug_text = yaml_text
            if structure is not None:

                def _annotate(match: re.Match[str]) -> str:
                    key = match.group(1)
                    try:
                        seq, bar, beat = decode_key(key)
                        t = structure.time_of(seq, bar, beat)
                        return f"{key} [t={t:.2f}s]:"
                    except (LLMFormatError, KeyError):
                        return f"{key} [t=?]:"

                debug_text = re.sub(rf"({KEY_PATTERN}):", _annotate, yaml_text)
            print("\n" + "=" * 80)
            print("EXTRACTED YAML TEXT (after slicing):")
            print(debug_text)
            print("=" * 80 + "\n")

        # Step 1: Extract the chunk between `choreography:` and `END` or end of file.
        match = re.search(r"choreography:\s*(.*?)(?:\s*END|$)", yaml_text, re.DOTALL)
        if not match:
            raise LLMFormatError(
                "Could not find a valid choreography in the YAML text. Make sure to start the "
                "choreography plan with the 'choreography' keyword."
            )
        choreography = match.group(1).strip()

        # Step 2: Strip line comments (everything after `#`).
        choreography = "\n".join(line.split("#")[0].strip() for line in choreography.splitlines())

        # Step 3: Parse `s<seq>b<bar>t<beat>: <action>` entries.
        entry_re = re.compile(rf"({KEY_PATTERN}):\s*(.*?)\s*(?={KEY_PATTERN}:|$)", re.DOTALL)
        choreography_steps: dict[tuple[int, int, int], str] = {}
        for entry in entry_re.findall(choreography):
            key_str, action_str = entry
            try:
                addr = decode_key(key_str)
            except LLMFormatError:
                raise
            choreography_steps[addr] = action_str.strip()

        if not choreography_steps:
            raise LLMFormatError(
                "No choreography entries parsed. Keys must be in s<seq>b<bar>t<beat> form, "
                "e.g. s1b1t1."
            )

        return dict(sorted(choreography_steps.items()))

    def _motion_primitives2time_and_pos(
        self, motion_primitives: dict, timestamps: NDArray, t_end: float
    ) -> tuple[NDArray, NDArray]:
        """Convert motion primitives to waypoints over forward-looking intervals.

        Each primitive plays from its own action time until the next action's time; the final
        primitive runs until ``t_end`` (the song's end). Drones hold their start positions
        until the first action.

        Returns:
            The motion primitive waypoint timings and positions.
        """
        waypoints = {}
        # TODO: Remove all conversions into cm
        swarm_pos = np.array(list(self.starting_pos.values())) * 100
        waypoints[0] = {i: p.copy() for i, p in enumerate(swarm_pos)}
        # Forward-looking intervals: primitive i plays [T_i, T_{i+1}], the last one to t_end.
        # _merge_motion_primitives reads tstart=timesteps[i-1], tend=timesteps[i] for key i.
        timesteps = np.concatenate((timestamps, [t_end]))
        motion_primitives = self._merge_motion_primitives(motion_primitives, timesteps)
        for motion_primitive in motion_primitives.values():
            for fn, args in zip(motion_primitive["fn"], motion_primitive["args"]):
                swarm_pos, _waypoints = self._primitive2waypoints(
                    fn, args, swarm_pos, motion_primitive["tstart"], motion_primitive["tend"]
                )
                for k, v in _waypoints.items():
                    waypoints[k] = v if k not in waypoints else waypoints[k] | v

        waypoints = self._fill_missing_waypoints(waypoints)
        waypoints = dicts2arrays(waypoints)
        pos = einops.rearrange(np.array(list(waypoints.values())), "t d c -> d t c")
        pos /= 100  # Convert back to meters. TODO: Remove all conversions
        t = np.tile(np.array(list(waypoints.keys())), (self.num_drones, 1))
        return t, pos

    def _fill_missing_waypoints(
        self, waypoints: dict[float, dict[int, NDArray]]
    ) -> dict[float, dict[int, NDArray]]:
        """Fill in missing waypoints.

        Some motion primitives operate on a subset of drones. Therefore, some drones will not have a
        waypoint at every timestep. We fill in the missing ones by copying over the previous
        timestep.
        """
        for i, waypoint in enumerate(waypoints.values()):
            # First time step must have all drones because we added the start positions at time 0
            if i == 0:
                assert all(d in waypoint for d in range(self.num_drones)), "Missing start positions"
                continue
            for drone_id in range(self.num_drones):
                if drone_id not in waypoint:
                    waypoint[drone_id] = list(waypoints.values())[i - 1][drone_id]
        return waypoints

    def _merge_motion_primitives(self, motion_primitives: dict, timesteps: NDArray) -> dict:
        """Merge and annotate motion primitives.

        Merge multiple motion primitives for a single timestep, add time information and add the
        time from PLAN motion_primitives to the previous function.
        """
        merged_motion_primitives = []
        # Filter out any PLAN motion_primitives that are at the end of the list. Make sure to not cut off
        # any other motion_primitives.
        if max(motion_primitives.keys()) >= len(timesteps):
            excess_primitives = [
                [list(d.keys())[0] for d in motion_primitives[i]]
                for i in motion_primitives
                if i >= len(timesteps)
            ]
            if not all("PLAN" in primitive for primitive in excess_primitives):
                raise LLMFormatError(
                    "Number of timesteps in output doesn't match the number of beats."
                )
            motion_primitives = {
                i: motion_primitives[i] for i in motion_primitives if i < len(timesteps)
            }
        for i in sorted(motion_primitives):
            fns = [list(d.keys())[0] for d in motion_primitives[i]]
            if i == 1 and "PLAN" in fns:
                raise LLMFormatError("PLAN can't be in the first step.")
            if "PLAN" in fns:
                merged_motion_primitives[-1]["tend"] = timesteps[i]
                merged_motion_primitives[-1]["steps"] += 1
                continue
            merged_motion_primitives.append(
                {
                    "fn": fns,
                    "args": [list(d.values())[0] for d in motion_primitives[i]],
                    "key": i,
                    "steps": 1,
                    "tstart": timesteps[i - 1],
                    "tend": timesteps[i],
                }
            )
        motion_primitives = {primitive["key"]: primitive for primitive in merged_motion_primitives}
        # Check that the motion primitives do not exceed the number of waypoints
        for motion_primitive in motion_primitives.values():
            if motion_primitive["key"] + motion_primitive["steps"] > len(timesteps):
                raise LLMFormatError(
                    (
                        f"Function {motion_primitive['fn']} at time {motion_primitive['key']} "
                        f"exceeds the number of allowed waypoints {len(timesteps)}"
                    )
                )
        return motion_primitives

    def _primitive2waypoints(
        self, fn_name: str, args: tuple, swarm_pos: dict, tstart: float, tend: float
    ) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
        """Convert a motion primitive to waypoint coordinates."""
        if fn_name == "PLAN":
            raise ValueError("PLAN should have been handled before")
        fn = primitive_by_name(fn_name)
        if motion_primitives_collection[fn_name]["n_args"] != len(args):
            raise LLMFormatError(f"Wrong number of arguments for {fn_name}")
        limits = {"lower": self.lim_lower, "upper": self.lim_upper}
        # We need to pass waypoints and swarm_pos because some motion primitives operate on a subset
        # of drones. Therefore, waypoints could contain positions for only some of the drones.
        # swarm_pos always tracks the current position of all drones. We also need the dictionary
        # instead of a list of positions in waypoints to track which drones have been moved.
        swarm_pos, waypoints = fn(args, swarm_pos, tstart, tend, limits)
        return swarm_pos, waypoints


def dicts2arrays(dict_of_dicts: dict[float, dict[int, NDArray]]) -> dict[float, NDArray]:
    """Convert a dictionary of dictionaries to a dictionary of arrays.

    Assumes that all inner dictionaries have the same keys.
    """
    dict_of_lists = {}
    for outer_key, inner_dict in dict_of_dicts.items():
        if inner_dict:
            dict_of_lists[outer_key] = [inner_dict[key] for key in sorted(inner_dict.keys())]
    homogeneous_len = len(list(dict_of_lists.values())[0])
    if not all(len(v) == homogeneous_len for v in dict_of_lists.values()):
        raise RuntimeError("Expected all lists to have the same length")
    return {k: np.array(v) for k, v in dict_of_lists.items()}


def _render_segments_table(structure: SongStructure) -> str:
    """Render a SongStructure as the multi-line block injected into the prompt.

    Args:
        structure: Song structure to describe.

    Returns:
        A newline-joined string with one indented line per segment, e.g.::

            segment 1: "intro" (0.00s - 12.30s) — 6 bars × 4 beats
            segment 2: "verse" (12.30s - 32.10s) — 10 bars × 4 beats
    """
    lines: list[str] = []
    for seg in structure.segments:
        n_bars = len(seg.bars)
        beats_per_bar = max((len(bar.beats) for bar in seg.bars), default=0)
        lines.append(
            f'  segment {seg.id}: "{seg.label}" '
            f"({seg.start_s:.2f}s - {seg.end_s:.2f}s) — "
            f"{n_bars} bars × {beats_per_bar} beats"
        )
    return "\n".join(lines)

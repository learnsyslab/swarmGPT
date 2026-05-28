"""A simple class to create and deploy a drone swarm."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cflib2 import Crazyflie, LinkContext
from cflib2.error import CrazyflieError, DisconnectedError, LinkError, TimeoutError
from cflib2.toc_cache import FileTocCache

os.environ["SCIPY_ARRAY_API"] = "1"
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping

    from drone_estimators.ros_nodes.ros2_connector import ROSConnector
    from numpy.typing import NDArray as Array
    from scipy.interpolate import BSpline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_DISCONNECT_ERRORS = (DisconnectedError, LinkError, TimeoutError)
_LIGHTHOUSE_DECK_PARAM = "deck.bcLighthouse4"
_POWER_CYCLE_BOOT_WAIT = 3.0


class DroneSwarm:
    """Connects, configures, and commands a Crazyflie swarm with cflib2."""

    def __init__(
        self,
        drones: dict[str, dict[str, Array]],
        ctrl_freq: float = 50,
        update_freq: float = 10,
        col_freq: float = 10,
        lighthouse: bool = True,
    ):
        """Create and connect a Crazyflie swarm.

        Args:
            drones: Dictionary of drones.toml entries including id, pos, and uri.
            ctrl_freq: Control frequency (Hz). Defaults to 50.
            update_freq: Frequency (Hz) of position updates sent to the drone. Defaults to 10.
            col_freq: Maximum frequency (Hz) of color updates. Defaults to 10.
            lighthouse: Whether to use lighthouse or mocap for localization. Defaults to True.
        """
        self.drones = drones
        self.ctrl_freq = ctrl_freq
        self.update_freq = update_freq
        self.col_freq = col_freq
        self.lighthouse = lighthouse

        self.uris = [d["uri"] for d in self.drones.values()]
        self.cfs: dict[str, Crazyflie] = {}
        self.active_uris: set[str] = set()
        self.context = LinkContext()
        self.toc_cache = FileTocCache("./cache")
        self.ros_connector: ROSConnector | None = None
        self._loop = asyncio.new_event_loop()
        self._closed = False

        if not lighthouse:
            from drone_estimators.ros_nodes.ros2_connector import ROSConnector

            self.ros_connector = ROSConnector(
                tf_names=[f"cf{int(d['uri'][-2:], 16)}" for d in self.drones.values()], timeout=10.0
            )

        try:
            self._run(self._connect())
            if self.lighthouse:
                self._run(self._check_lighthouse_decks())
            self.reset()
        except BaseException:
            if self.cfs:
                try:
                    self._run(self._disconnect())
                except Exception as exc:
                    logger.error(f"Disconnecting after initialization failure failed: {exc}")
            self._loop.close()
            if self.ros_connector is not None:
                self.ros_connector.close()
            raise
        logger.info("init done")

    def get_obs(self, uri: str) -> dict[str, Array]:
        """Generate the observation for a drone using mocap or lighthouse."""
        return self._run(self._read_observation(uri))

    def missing_uris(self) -> list[str]:
        """Return configured URIs that are not currently active."""
        return [uri for uri in self.uris if uri not in self.active_uris]

    def is_active(self, uri: str) -> bool:
        """Return whether a configured URI is still active."""
        return uri in self.active_uris

    def takeoff(self, height: float = 1.5, duration: float = 3.0):
        """Take off the drones to a given height over a given duration."""

        async def _takeoff(uri: str) -> None:
            await self._takeoff_one(uri, height, duration)

        self._run(self._parallel_by_uri("Taking off", self.uris, _takeoff))

    def land(self, height: float = 0.0, duration: float = 3.0):
        """Land the drones at a given height over a given duration."""

        async def _land(uri: str) -> None:
            await self._land_one(uri, height, duration)

        self._run(self._parallel_by_uri("Landing", self.uris, _land))

    def goto(self, pos: dict[str, list], duration: float = 3.0):
        """Execute a go to command for all drones by linearly interpolating references.

        Args:
            pos: Position+Yaw references in the form {'uri1': [pos], ...}.
            duration: Duration of the connection in seconds.
        """
        self._validate_required_uris("pos", pos)
        for uri, target in pos.items():
            if len(target) != 1:
                raise ValueError(f"pos[{uri!r}] must contain exactly one target.")

        async def _goto(uri: str) -> None:
            target = pos[uri]
            await self._goto_one(uri, target[0], duration)

        self._run(self._parallel_by_uri("Go to", self.uris, _goto))

    def execute_choreography(
        self,
        choreography: dict[str, BSpline],
        t_end: float,
        *,
        color_top: dict[str, dict[float, Array]] | None = None,
        color_bot: dict[str, dict[float, Array]] | None = None,
    ):
        """Execute a choreography with position, orientation, and light commands.

        Args:
            choreography: Reference in the form of a 3d spline.
            t_end: End time of the choreography.
            color_top: Top deck color cues in the form {uri: {time: wrgb}}.
            color_bot: Bottom deck color cues in the form {uri: {time: wrgb}}.
        """
        self._validate_required_uris("choreography", choreography)
        if color_top is None and color_bot is None:
            logger.warning("No colors provided for choreography.")
        self._validate_known_uris("color_top", color_top or {})
        self._validate_known_uris("color_bot", color_bot or {})

        async def _execute(uri: str) -> None:
            await self._execute_one(
                uri,
                choreography[uri],
                t_end,
                (color_top or {}).get(uri, {}),
                (color_bot or {}).get(uri, {}),
            )

        self._run(self._parallel_by_uri("Choreography execution", self.uris, _execute))

    def apply_colors(self, color_top: dict[str, Array] | None, color_bot: dict[str, Array] | None):
        """Apply colors to the drones.

        Args:
            color_top: Top deck colors in the form {uri: wrgb}.
            color_bot: Bottom deck colors in the form {uri: wrgb}.
        """
        if color_top is None:
            color_top = dict.fromkeys(self.uris, np.zeros(4))
        if color_bot is None:
            color_bot = dict.fromkeys(self.uris, np.zeros(4))
        self._validate_known_uris("color_top", color_top)
        self._validate_known_uris("color_bot", color_bot)

        async def _apply_colors(uri: str) -> None:
            if uri in color_top:
                await self._apply_drone_color(uri, color_top[uri], "top")
            if uri in color_bot:
                await self._apply_drone_color(uri, color_bot[uri], "bot")

        self._run(self._parallel_by_uri("Applying colors", self.uris, _apply_colors))

    def set_param(self, param: str, value: float):
        """Set a Crazyflie parameter on all active drones.

        Args:
            param: Parameter name in ``group.name`` format.
            value: Value to set.
        """

        async def _set_param(uri: str) -> None:
            await self._set_param_one(uri, param, value)

        self._run(self._parallel_by_uri(f"Setting parameter {param}", self.uris, _set_param))

    def emergency_stop(self, id: int | None = None):
        """Send an emergency stop signal to one (id) or all drones (default)."""
        if id is not None:
            raise NotImplementedError("Sending emergency stop to one drone not implemented.")

        self._run(self._parallel_by_uri("Emergency stop", self.uris, self._emergency_stop_one))

    def reset(self):
        """Reset all active drones."""
        self._run(self._parallel_by_uri("Resetting", self.uris, self._reset_one))

    def close(self):
        """Close the swarm and ROS connection."""
        if self._closed:
            return
        self._closed = True
        try:
            self._run(self._close())
        finally:
            self._loop.close()
            if self.ros_connector is not None:
                self.ros_connector.close()

    def _run(self, coroutine: Awaitable[Any]) -> Any:
        """Run a cflib2 coroutine on the swarm event loop."""
        return self._loop.run_until_complete(coroutine)

    def _cf(self, uri: str) -> Crazyflie:
        if uri not in self.active_uris:
            raise RuntimeError(f"Drone {uri} is not active.")
        try:
            return self.cfs[uri]
        except KeyError as exc:
            raise RuntimeError(f"Drone {uri} is not connected.") from exc

    def _validate_required_uris(self, name: str, mapping: Mapping[str, object]) -> None:
        expected = set(self.uris)
        actual = set(mapping)
        missing = expected - actual
        unknown = actual - expected
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if unknown:
                details.append(f"unknown {sorted(unknown)}")
            raise ValueError(f"{name} must contain all configured drone URIs: {'; '.join(details)}")

    def _validate_known_uris(self, name: str, mapping: Mapping[str, object]) -> None:
        unknown = set(mapping) - set(self.uris)
        if unknown:
            raise ValueError(f"{name} contains unknown drone URIs: {sorted(unknown)}")

    async def _connect(self) -> None:
        await self._power_cycle()
        await asyncio.sleep(_POWER_CYCLE_BOOT_WAIT)

        results = await asyncio.gather(
            *[Crazyflie.connect_from_uri(self.context, uri, self.toc_cache) for uri in self.uris],
            return_exceptions=True,
        )
        failures = []
        for uri, result in zip(self.uris, results, strict=True):
            if isinstance(result, BaseException):
                failures.append(f"{uri}: {result}")
                continue
            self.cfs[uri] = result

        if failures:
            await asyncio.gather(
                *[cf.disconnect() for cf in self.cfs.values()], return_exceptions=True
            )
            self.cfs.clear()
            raise RuntimeError(f"Connecting to Crazyflies failed: {'; '.join(failures)}")

        self.active_uris = set(self.cfs)

        if not self.active_uris:
            raise RuntimeError("No Crazyflies connected.")

    async def _check_lighthouse_decks(self) -> None:
        results = await asyncio.gather(
            *[self._cf(uri).param().get(_LIGHTHOUSE_DECK_PARAM) for uri in self.uris],
            return_exceptions=True,
        )
        failures = []
        for uri, result in zip(self.uris, results, strict=True):
            if isinstance(result, BaseException):
                failures.append(f"{uri}: could not read {_LIGHTHOUSE_DECK_PARAM}: {result}")
                continue
            if result != 1:
                failures.append(f"{uri}: {_LIGHTHOUSE_DECK_PARAM}={result!r}")

        if failures:
            raise RuntimeError(
                "Lighthouse deck check failed. Expected "
                f"{_LIGHTHOUSE_DECK_PARAM}=1 for every drone: {'; '.join(failures)}"
            )

    async def _power_cycle(self) -> None:
        async def _cycle(uri: str) -> None:
            try:
                await Crazyflie.power_off_stm32_domain(self.context, uri)
                await asyncio.sleep(0.1)
                await Crazyflie.power_on_stm32_domain(self.context, uri)
            except CrazyflieError as exc:
                logger.warning(f"Power cycling {uri} failed: {exc}")

        await asyncio.gather(*[_cycle(uri) for uri in self.uris])

    async def _parallel_by_uri(
        self, action_name: str, uris: Iterable[str], action: Callable[[str], Awaitable[None]]
    ) -> None:
        target_uris = [uri for uri in uris if uri in self.active_uris and uri in self.cfs]

        results = await asyncio.gather(
            *[action(uri) for uri in target_uris], return_exceptions=True
        )

        for uri, result in zip(target_uris, results, strict=True):
            if not isinstance(result, BaseException):
                continue
            if isinstance(result, _DISCONNECT_ERRORS):
                self.active_uris.discard(uri)
                logger.error(f"{uri} disconnected or unreachable. {action_name} failed: {result}")
            else:
                logger.error(f"{action_name} failed for {uri}: {result}")

    async def _read_observation(self, uri: str) -> dict[str, Array]:
        if self.lighthouse:
            return await self._read_lighthouse_observation(uri)

        assert self.ros_connector is not None, "Mocap observations require lighthouse=False."
        drone_name = f"cf{int(uri[-2:], 16):02d}"
        obs = {
            "pos": self.ros_connector.pos[drone_name],
            "quat": self.ros_connector.quat[drone_name],
            # "is_outdated": ros_connector.is_outdated[drone_name],  # estimator-only
            # TODO add is_outdated feature to ros_connector for tfs
            "is_outdated": False,
        }
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        return obs

    async def _read_lighthouse_observation(self, uri: str) -> dict[str, Array]:
        cf = self._cf(uri)
        block = await cf.log().create_block()
        await block.add_variable("stateEstimate.x")
        await block.add_variable("stateEstimate.y")
        await block.add_variable("stateEstimate.z")
        await block.add_variable("stateEstimate.roll")
        await block.add_variable("stateEstimate.pitch")
        await block.add_variable("stateEstimate.yaw")

        stream = await block.start(10)
        try:
            data = await stream.next()
            values = data.data
            pos = np.array(
                [values["stateEstimate.x"], values["stateEstimate.y"], values["stateEstimate.z"]]
            )
            rpy = np.deg2rad(
                [
                    values["stateEstimate.roll"],
                    values["stateEstimate.pitch"],
                    values["stateEstimate.yaw"],
                ]
            )
            return {
                "pos": pos,
                "quat": R.from_euler("xyz", rpy).as_quat(),
                "rpy": rpy,
                "is_outdated": False,
            }
        finally:
            await stream.stop()

    async def _apply_drone_color(
        self,
        uri: str,
        wrgb: Array = np.array([0, 0, 0, 0]),
        deck: Literal["top", "bot", "both"] = "both",
    ) -> None:
        assert np.all((wrgb >= 0) & (wrgb <= 255)), (
            f"Valid range for wrgb values is [0,255], was {wrgb}"
        )
        w, r, g, b = int(wrgb[0]), int(wrgb[1]), int(wrgb[2]), int(wrgb[3])
        color = int(f"0x{w:02x}{r:02x}{g:02x}{b:02x}", 16)
        param = self._cf(uri).param()
        if deck == "top" or deck == "both":
            await param.set("colorLedTop.wrgb8888", color)
        if deck == "bot" or deck == "both":
            await param.set("colorLedBot.wrgb8888", color)

    async def _set_param_one(self, uri: str, param_name: str, value: float) -> None:
        await self._cf(uri).param().set(param_name, value)

    async def _reset_one(self, uri: str) -> None:
        cf = self._cf(uri)
        param = cf.param()
        # Estimator setting;  1: complementary, 2: kalman
        await param.set("stabilizer.estimator", 2)
        # Enable/disable tumble control. Required 0 for aggressive maneuvers.
        await param.set("supervisor.tmblChckEn", 1)
        # Choose controller: 1: PID; 2: Mellinger.
        await param.set("stabilizer.controller", 2)
        await param.set("led.bitmask", 128)  # turn off all LEDs
        await asyncio.sleep(0.1)

        if not self.lighthouse:
            obs = await self._read_observation(uri)
            await param.set("kalman.initialX", float(obs["pos"][0]))
            await param.set("kalman.initialY", float(obs["pos"][1]))
            await param.set("kalman.initialZ", float(obs["pos"][2]))
            await param.set("kalman.initialYaw", float(obs["rpy"][2]))

        await param.set("kalman.resetEstimation", 1)
        await asyncio.sleep(0.1)
        await param.set("kalman.resetEstimation", 0)
        await cf.platform().send_arming_request(do_arm=True)
        await asyncio.sleep(0.8)

    async def _send_external_pose(self, uri: str) -> None:
        if self.lighthouse:
            return
        obs = await self._read_observation(uri)
        await (
            self._cf(uri)
            .localization()
            .external_pose()
            .send_external_pose(
                pos=np.asarray(obs["pos"], dtype=float).tolist(),
                quat=np.asarray(obs["quat"], dtype=float).tolist(),
            )
        )

    async def _takeoff_one(self, uri: str, height: float, duration: float) -> None:
        cf = self._cf(uri)
        await self._send_external_pose(uri)
        commander = cf.commander()
        await commander.send_stop_setpoint()
        await commander.send_notify_setpoint_stop(0)
        await cf.param().set("commander.enHighLevel", 1)
        await cf.high_level_commander().take_off(height, None, duration, None)
        await self._update_external_pose_during(uri, duration)

    async def _land_one(self, uri: str, height: float, duration: float) -> None:
        cf = self._cf(uri)
        commander = cf.commander()
        await commander.send_stop_setpoint()
        await commander.send_notify_setpoint_stop(0)
        await cf.param().set("commander.enHighLevel", 1)
        high_level_commander = cf.high_level_commander()
        await high_level_commander.land(height, None, duration, None)
        await self._update_external_pose_during(uri, duration)
        await high_level_commander.stop(None)

    async def _update_external_pose_during(self, uri: str, duration: float) -> None:
        end_time = asyncio.get_running_loop().time() + duration
        while asyncio.get_running_loop().time() < end_time:
            await self._send_external_pose(uri)
            await asyncio.sleep(1 / self.update_freq)

    async def _goto_one(self, uri: str, pos: Array, duration: float) -> None:
        cf = self._cf(uri)
        await cf.param().set("commander.enHighLevel", 0)
        obs = await self._read_observation(uri)
        pos_start = np.array([*obs["pos"], np.degrees(obs["rpy"][2])])
        pos_goal = np.asarray(pos, dtype=float)
        ref = interp1d([0.0, duration], [pos_start, pos_goal], axis=0)
        await self._stream_reference(uri, duration, lambda t: np.asarray(ref(t), dtype=float))

    async def _execute_one(
        self,
        uri: str,
        choreography: BSpline,
        t_end: float,
        color_top: dict[float, Array],
        color_bot: dict[float, Array],
    ) -> None:
        cf = self._cf(uri)
        await cf.param().set("commander.enHighLevel", 0)
        await self._stream_reference(
            uri,
            t_end,
            lambda t: np.asarray([*choreography(t), 0.0], dtype=float),
            color_top,
            color_bot,
        )

    async def _stream_reference(
        self,
        uri: str,
        duration: float,
        reference: Callable[[float], Array],
        color_top: dict[float, Array] | None = None,
        color_bot: dict[float, Array] | None = None,
    ) -> None:
        commander = self._cf(uri).commander()
        t_est = -np.inf
        top_cues = sorted((float(t), wrgb) for t, wrgb in (color_top or {}).items())
        bot_cues = sorted((float(t), wrgb) for t, wrgb in (color_bot or {}).items())
        i_next_top = 0
        i_next_bot = 0
        period = 1 / self.ctrl_freq
        start_time = asyncio.get_running_loop().time()
        next_tick = start_time

        while (t_cur := asyncio.get_running_loop().time() - start_time) < duration:
            if not self.lighthouse and t_cur - t_est >= 1 / self.update_freq:
                await self._send_external_pose(uri)
                t_est = t_cur

            await commander.send_setpoint_position(*reference(t_cur))

            while i_next_top < len(top_cues) and t_cur >= top_cues[i_next_top][0]:
                await self._apply_drone_color(uri, top_cues[i_next_top][1], "top")
                i_next_top += 1
            while i_next_bot < len(bot_cues) and t_cur >= bot_cues[i_next_bot][0]:
                await self._apply_drone_color(uri, bot_cues[i_next_bot][1], "bot")
                i_next_bot += 1

            next_tick += period
            await asyncio.sleep(max(0.0, next_tick - asyncio.get_running_loop().time()))

    async def _emergency_stop_one(self, uri: str) -> None:
        await self._cf(uri).localization().emergency().send_emergency_stop()

    async def _shutdown_leds_one(self, uri: str) -> None:
        await self._cf(uri).param().set("led.bitmask", 0)  # turn on all LEDs to indicate shutdown
        await self._apply_drone_color(uri, np.zeros(4), "both")

    async def _close(self) -> None:
        active_uris = [uri for uri in self.uris if uri in self.active_uris]
        if active_uris:
            try:
                await self._parallel_by_uri("Emergency stop", active_uris, self._emergency_stop_one)
                await asyncio.sleep(0.1)
                await self._parallel_by_uri("Shutdown LEDs", active_uris, self._shutdown_leds_one)
                await asyncio.sleep(0.2)
            except RuntimeError as exc:
                logger.warning(f"Shutdown failed: {exc}")

        await self._disconnect()

    async def _disconnect(self) -> None:
        disconnect_results = await asyncio.gather(
            *[cf.disconnect() for cf in self.cfs.values()], return_exceptions=True
        )
        for uri, result in zip(self.cfs.keys(), disconnect_results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Disconnecting {uri} failed: {result}")

        self.active_uris.clear()

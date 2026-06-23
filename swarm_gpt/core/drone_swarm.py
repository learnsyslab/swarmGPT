"""A simple class to create and deploy a drone swarm."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import threading
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from cflib2 import Crazyflie, LinkContext
from cflib2.error import CrazyflieError, DisconnectedError, LinkError, TimeoutError
from cflib2.toc_cache import FileTocCache

os.environ["SCIPY_ARRAY_API"] = "1"
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping
    from concurrent.futures import Future
    from multiprocessing.synchronize import Event

    from drone_estimators.ros_nodes.ros2_connector import ROSConnector
    from numpy.typing import NDArray as Array
    from scipy.interpolate import BSpline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_DISCONNECT_ERRORS = (DisconnectedError, LinkError, TimeoutError)
_LIGHTHOUSE_DECK_PARAM = "deck.bcLighthouse4"
_POWER_CYCLE_BOOT_WAIT = 3.0
_CommanderLevel = Literal["low", "high"]


class EmergencyStopActive(RuntimeError):
    """Raised when a motion command is issued after the emergency-stop latch is set."""


class DroneSwarm:
    """Connects, configures, and commands a Crazyflie swarm with cflib2."""

    def __init__(
        self,
        drones: dict[str, dict[str, Array]],
        ctrl_freq: float = 50,
        update_freq: float = 20,
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
        self._commander_levels: dict[str, _CommanderLevel | None] = dict.fromkeys(self.uris, None)
        self.context = LinkContext()
        self.toc_cache = FileTocCache("./cache")
        self.ros_connector: ROSConnector | None = None
        self._loop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None
        self._estimator_stop_event: Event | None = None
        self._estimator_future: Future[None] | None = None
        self._closed = False
        self._estop = threading.Event()

        if not lighthouse:
            from drone_estimators.ros_nodes.ros2_connector import ROSConnector

            self.ros_connector = ROSConnector(
                tf_names=[f"cf{int(d['uri'][-2:], 16)}" for d in self.drones.values()], timeout=10.0
            )

        try:
            self._run(self._connect())
            if self.lighthouse:
                self._run(self._check_lighthouse_decks())
            else:
                self._start_estimator_updater()
            self.reset()
        except BaseException:
            if self._estimator_stop_event is not None:
                self._estimator_stop_event.set()
            if self._estimator_future is not None:
                try:
                    self._estimator_future.result(timeout=max(1.0, 2 / self.update_freq))
                except Exception as exc:
                    logger.warning(f"Stopping estimator updater failed: {exc}")
            if self.cfs:
                try:
                    self._run(self._disconnect())
                except Exception as exc:
                    logger.error(f"Disconnecting after initialization failure failed: {exc}")
            self._stop_loop_thread()
            self._loop.close()
            if self.ros_connector is not None:
                self.ros_connector.close()
            raise
        logger.info("init done")

    def get_obs(self, uri: str) -> dict[str, Array]:
        """Generate the observation for a drone using mocap or lighthouse."""
        obs = self._run(self._parallel_by_uri("Reading observation", [uri], self._read_observation))
        return obs[0]

    def missing_uris(self) -> list[str]:
        """Return configured URIs that are not currently active."""
        return [uri for uri in self.uris if uri not in self.active_uris]

    def is_active(self, uri: str) -> bool:
        """Return whether a configured URI is still active."""
        return uri in self.active_uris

    def takeoff(self, height: float = 1.5, duration: float = 3.0):
        """Take off the drones to a given height over a given duration."""
        self._ensure_not_estopped()

        async def _takeoff(uri: str) -> None:
            cf = self._cf(uri)
            await self._change_commander_level(uri, "high")
            await cf.high_level_commander().take_off(height, None, duration, None)
            await self._estop_guarded_sleep(uri, duration)

        self._run(self._parallel_by_uri("Taking off", self.uris, _takeoff, timeout=duration + 1.0))

    def land(self, height: float = 0.0, duration: float = 3.0):
        """Land the drones at a given height over a given duration."""
        self._ensure_not_estopped()

        async def _land(uri: str) -> None:
            cf = self._cf(uri)
            await self._change_commander_level(uri, "high")
            high_level_commander = cf.high_level_commander()
            await high_level_commander.land(height, None, duration, None)
            if await self._estop_guarded_sleep(uri, duration):
                return
            await high_level_commander.stop(None)

        self._run(self._parallel_by_uri("Landing", self.uris, _land, timeout=duration + 1.0))

    def goto(self, target: dict[str, list], duration: float = 3.0):
        """Execute a high-level goto command for all drones.

        Args:
            target: Position+Yaw references in the form {'uri1': [target], ...}.
            duration: Duration of the motion in seconds.
        """
        self._ensure_not_estopped()
        self._validate_required_uris("pos", target)
        for uri, setpoint in target.items():
            if len(setpoint) != 4:
                raise ValueError(f"pos[{uri!r}] must contain exactly four elements.")

        async def _goto(uri: str) -> None:
            cf = self._cf(uri)
            await self._change_commander_level(uri, "high")
            await cf.high_level_commander().go_to(
                *target[uri], duration, relative=False, linear=True, group_mask=None
            )
            await self._estop_guarded_sleep(uri, duration)

        self._run(self._parallel_by_uri("Goto", self.uris, _goto, timeout=duration + 1.0))

    def setpoint(self, target: dict[str, list]):
        """Send one position+yaw setpoint to all drones and return.

        Args:
            target: Position+Yaw references in the form {'uri1': [target], ...}.
        """
        self._ensure_not_estopped()
        self._validate_required_uris("pos", target)
        for uri, setpoint in target.items():
            if len(setpoint) != 4:
                raise ValueError(f"pos[{uri!r}] must contain exactly four elements.")

        async def _setpoint(uri: str) -> None:
            await self._change_commander_level(uri, "low")
            await self._cf(uri).commander().send_setpoint_position(*target[uri])

        self._run(self._parallel_by_uri("Setpoint", self.uris, _setpoint, timeout=0.5))

    def execute_choreography(
        self,
        choreography: dict[str, BSpline],
        t_end: float,
        *,
        color_top: dict[str, dict[float, Array]] = {},
        color_bot: dict[str, dict[float, Array]] = {},
    ):
        """Execute a choreography with position, orientation, and light commands.

        Args:
            choreography: Reference in the form of a 3d spline.
            t_end: End time of the choreography.
            color_top: Top deck color cues in the form {uri: {time: wrgb}}.
            color_bot: Bottom deck color cues in the form {uri: {time: wrgb}}.
        """
        self._ensure_not_estopped()
        self._validate_required_uris("choreography", choreography)
        if not color_top and not color_bot:
            logger.warning("No colors provided for choreography.")
        self._validate_known_uris("color_top", color_top)
        self._validate_known_uris("color_bot", color_bot)

        async def _execute(uri: str) -> None:
            await self._change_commander_level(uri, "low")
            await self._stream_reference(
                uri,
                t_end,
                lambda t: np.asarray([*choreography[uri](t), 0.0], dtype=float),
                color_top.get(uri, {}),
                color_bot.get(uri, {}),
            )

        self._run(
            self._parallel_by_uri(
                "Choreography execution", self.uris, _execute, timeout=t_end + 1.0
            )
        )

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

        self._run(self._parallel_by_uri("Applying colors", self.uris, _apply_colors, timeout=0.5))

    def set_param(self, param: str, value: float):
        """Set a Crazyflie parameter on all active drones.

        Args:
            param: Parameter name in ``group.name`` format.
            value: Value to set.
        """

        async def _set_param(uri: str) -> None:
            await self._cf(uri).param().set(param, value)

        self._run(
            self._parallel_by_uri(f"Setting parameter {param}", self.uris, _set_param, timeout=0.5)
        )

    def emergency_stop(self, uri: str | None = None):
        """Send an emergency stop signal to one URI or all drones (default).

        Sets a latch that halts the choreography stream and blocks further motion
        commands so fresh setpoints cannot override the motor cut.
        """
        self._estop.set()
        if uri is None:
            uris = self.uris
        else:
            self._validate_known_uris("uri", {uri: None})
            uris = [uri]
        # In-flight loop coroutines self-cut on the latch; this also sends the packet directly
        # to cover idle windows. Bounded so the caller never blocks on a loop that has stopped.
        coro = self._parallel_by_uri("Emergency stop", uris, self._emergency_stop, timeout=0.5)
        try:
            if self._loop_thread is not None or self._loop.is_running():
                asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=1.0)
            else:
                self._loop.run_until_complete(coro)
        except Exception as exc:
            logger.error(f"Emergency stop encountered errors: {exc}")

    def reset(self):
        """Reset all active drones."""

        async def _reset(uri: str) -> None:
            cf = self._cf(uri)
            param = cf.param()
            # Estimator setting;  1: complementary, 2: kalman
            await param.set("stabilizer.estimator", 2)
            # Enable/disable tumble control. Required 0 for aggressive maneuvers.
            await param.set("supervisor.tmblChckEn", 1)
            # Choose controller: 1: PID; 2: Mellinger.
            await param.set("stabilizer.controller", 2)
            await param.set("led.bitmask", 128)  # turn off all indicator LEDs
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
            await self._change_commander_level(uri, "high")
            await asyncio.sleep(0.8)

        self._run(self._parallel_by_uri("Resetting", self.uris, _reset))

    def close(self):
        """Close the swarm and ROS connection."""
        if self._closed:
            return
        self._closed = True

        async def _shutdown_leds(uri: str) -> None:
            await self._cf(uri).param().set("led.bitmask", 0)  # turn on all indicator LEDs
            await self._apply_drone_color(uri, np.zeros(4), "both")

        async def _close() -> None:
            active_uris = [uri for uri in self.uris if uri in self.active_uris]
            if active_uris:
                try:
                    await self._parallel_by_uri(
                        "Emergency stop", active_uris, self._emergency_stop, timeout=0.5
                    )
                    await asyncio.sleep(0.1)
                    await self._parallel_by_uri(
                        "Shutdown LEDs", active_uris, _shutdown_leds, timeout=0.5
                    )
                    await asyncio.sleep(0.2)
                except RuntimeError as exc:
                    logger.warning(f"Shutdown failed: {exc}")

            await self._disconnect()

        try:
            if self._estimator_stop_event is not None:
                self._estimator_stop_event.set()
            if self._estimator_future is not None:
                try:
                    self._estimator_future.result(timeout=max(1.0, 2 / self.update_freq))
                except Exception as exc:
                    logger.warning(f"Stopping estimator updater failed: {exc}")
            self._run(_close())
        finally:
            self._estimator_future = None
            self._estimator_stop_event = None
            self._stop_loop_thread()
            self._loop.close()
            if self.ros_connector is not None:
                self.ros_connector.close()

    def _ensure_not_estopped(self) -> None:
        """Block forward-motion commands once the emergency-stop latch is set."""
        if self._estop.is_set():
            raise EmergencyStopActive("Swarm is emergency-stopped; motion commands are blocked.")

    async def _estop_guarded_sleep(self, uri: str, duration: float) -> bool:
        """Wait up to ``duration``, cutting ``uri``'s motors from the loop if the latch trips.

        Returns True if the latch tripped (motors cut), False if it slept the full duration.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration
        while (remaining := deadline - loop.time()) > 0:
            if self._estop.is_set():
                await self._emergency_stop(uri)
                return True
            await asyncio.sleep(min(0.02, remaining))
        return False

    def _run(self, coroutine: Awaitable[Any]) -> Any:
        """Run a cflib2 coroutine on the swarm event loop.

        Dispatches cross-thread when the loop runs in another thread or is already running,
        so emergency stops from the request or signal-handler threads still reach the swarm.
        """
        if self._loop_thread is not None or self._loop.is_running():
            return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()
        return self._loop.run_until_complete(coroutine)

    def _start_estimator_updater(self) -> None:
        """Start continuous mocap updates after the radio links are connected."""
        ctx = mp.get_context("spawn")
        self._estimator_stop_event = ctx.Event()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="drone-swarm-event-loop", daemon=True
        )
        self._loop_thread.start()
        self._estimator_future = asyncio.run_coroutine_threadsafe(
            self._update_estimators(self._estimator_stop_event), self._loop
        )

    def _stop_loop_thread(self) -> None:
        """Stop the persistent event loop used by the estimator updater."""
        if self._loop_thread is None:
            return

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._loop_thread = None

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
        async def _power_cycle(uri: str) -> None:
            try:
                await Crazyflie.power_off_stm32_domain(self.context, uri)
                await asyncio.sleep(0.1)
                await Crazyflie.power_on_stm32_domain(self.context, uri)
            except CrazyflieError as exc:
                logger.warning(f"Power cycling {uri} failed: {exc}")

        await asyncio.gather(*[_power_cycle(uri) for uri in self.uris])
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
                "Lighthouse deck check failed. Are you using lighthouse? Expected "
                f"{_LIGHTHOUSE_DECK_PARAM}=1 for every drone: {'; '.join(failures)}"
            )

    async def _parallel_by_uri(
        self,
        action_name: str,
        uris: Iterable[str],
        action: Callable[[str], Awaitable[None]],
        timeout: float | None = None,
    ) -> None:
        target_uris = [uri for uri in uris if uri in self.active_uris and uri in self.cfs]

        async with asyncio.timeout(timeout):
            results = await asyncio.gather(
                *[action(uri) for uri in target_uris], return_exceptions=True
            )

        for uri, result in zip(target_uris, results, strict=True):
            if not isinstance(result, BaseException):
                continue
            if isinstance(result, _DISCONNECT_ERRORS):
                self.active_uris.discard(uri)
                self._commander_levels[uri] = None
                logger.error(f"{uri} disconnected or unreachable. {action_name} failed: {result}")
            else:
                logger.error(f"{action_name} failed for {uri}: {result}")
        return results

    async def _emergency_stop(self, uri: str) -> None:
        await self._cf(uri).localization().emergency().send_emergency_stop()

    async def _change_commander_level(self, uri: str, level: _CommanderLevel) -> None:
        """Switch commander level only when local state expects a different mode."""
        if self._commander_levels.get(uri) == level:
            return

        cf = self._cf(uri)
        if level == "high":
            await cf.commander().send_notify_setpoint_stop(0)
            # await asyncio.sleep(0.05)  # Might be necessary when firmware crashes during level switch

        await cf.param().set("commander.enHighLevel", int(level == "high"))
        self._commander_levels[uri] = level

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
        except Exception as exc:
            logger.error(f"Reading observation from {uri} failed: {exc}")
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

    async def _update_estimators(self, stop_event: Event) -> None:
        """Continuously send mocap poses to every active Crazyflie estimator."""
        assert self.ros_connector is not None, "Estimator updates require lighthouse=False."
        period = 1 / self.update_freq
        next_tick = asyncio.get_running_loop().time()
        while not stop_event.is_set():
            # These properties already copy the synchronized ROS arrays into fresh NumPy snapshots.
            positions = self.ros_connector.pos
            quaternions = self.ros_connector.quat

            async def _update_estimator(uri: str) -> None:
                drone_name = f"cf{int(uri[-2:], 16):02d}"
                await (
                    self._cf(uri)
                    .localization()
                    .external_pose()
                    .send_external_pose(
                        pos=positions[drone_name].tolist(), quat=quaternions[drone_name].tolist()
                    )
                )

            await self._parallel_by_uri("Updating estimators", self.uris, _update_estimator)
            next_tick += period
            await asyncio.sleep(max(0.0, next_tick - asyncio.get_running_loop().time()))

    async def _stream_reference(
        self,
        uri: str,
        duration: float,
        reference: Callable[[float], Array],
        color_top: dict[float, Array] | None = None,
        color_bot: dict[float, Array] | None = None,
    ) -> None:
        commander = self._cf(uri).commander()
        top_cues = sorted((float(t), wrgb) for t, wrgb in (color_top or {}).items())
        bot_cues = sorted((float(t), wrgb) for t, wrgb in (color_bot or {}).items())
        i_next_top = 0
        i_next_bot = 0
        period = 1 / self.ctrl_freq
        color_period = 1 / self.col_freq
        start_time = asyncio.get_running_loop().time()
        next_tick = start_time
        t_col = -np.inf

        while (t_cur := asyncio.get_running_loop().time() - start_time) < duration:
            if self._estop.is_set():
                await self._emergency_stop(uri)
                return
            await commander.send_setpoint_position(*reference(t_cur))

            if t_cur - t_col >= color_period:
                if i_next_top < len(top_cues) and t_cur >= top_cues[i_next_top][0]:
                    await self._apply_drone_color(uri, top_cues[i_next_top][1], "top")
                    i_next_top += 1
                if i_next_bot < len(bot_cues) and t_cur >= bot_cues[i_next_bot][0]:
                    await self._apply_drone_color(uri, bot_cues[i_next_bot][1], "bot")
                    i_next_bot += 1
                t_col = t_cur

            next_tick += period
            await asyncio.sleep(max(0.0, next_tick - asyncio.get_running_loop().time()))

    async def _disconnect(self) -> None:
        disconnect_results = await asyncio.gather(
            *[cf.disconnect() for cf in self.cfs.values()], return_exceptions=True
        )
        for uri, result in zip(self.cfs.keys(), disconnect_results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Disconnecting {uri} failed: {result}")

        self.active_uris.clear()

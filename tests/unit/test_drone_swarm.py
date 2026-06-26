import asyncio
import threading
from typing import Any

import numpy as np
import pytest
from cflib2.error import DisconnectedError

from swarm_gpt.core.drone_swarm import DroneSwarm


class FakeParam:
    def __init__(self) -> None:
        self.values: list[tuple[str, Any]] = []

    async def set(self, name: str, value: Any) -> None:
        self.values.append((name, value))


class FakeCommander:
    def __init__(self) -> None:
        self.setpoints: list[tuple[float, ...]] = []

    async def send_setpoint_position(self, *setpoint: float) -> None:
        self.setpoints.append(setpoint)


class FakeExternalPose:
    def __init__(self) -> None:
        self.sent: list[tuple[list[float], list[float]]] = []

    async def send_external_pose(self, *, pos: list[float], quat: list[float]) -> None:
        self.sent.append((pos, quat))


class FakeLocalization:
    def __init__(self) -> None:
        self.fake_external_pose = FakeExternalPose()

    def external_pose(self) -> FakeExternalPose:
        return self.fake_external_pose


class FakeCrazyflie:
    def __init__(self) -> None:
        self.fake_param = FakeParam()
        self.fake_commander = FakeCommander()
        self.fake_localization = FakeLocalization()

    def param(self) -> FakeParam:
        return self.fake_param

    def commander(self) -> FakeCommander:
        return self.fake_commander

    def localization(self) -> FakeLocalization:
        return self.fake_localization


class FakeROSConnector:
    def __init__(
        self,
        positions: dict[str, list[float]],
        quaternions: dict[str, list[float]],
        stop_event: threading.Event,
    ) -> None:
        self.positions = positions
        self.quaternions = quaternions
        self.stop_event = stop_event
        self.pos_reads = 0
        self.quat_reads = 0

    @property
    def pos(self) -> dict[str, np.ndarray]:
        self.pos_reads += 1
        return {name: np.asarray(pos) for name, pos in self.positions.items()}

    @property
    def quat(self) -> dict[str, np.ndarray]:
        self.quat_reads += 1
        self.stop_event.set()
        return {name: np.asarray(quat) for name, quat in self.quaternions.items()}


def make_swarm(uris: list[str]) -> DroneSwarm:
    swarm = object.__new__(DroneSwarm)
    swarm.uris = uris
    swarm.cfs = {uri: FakeCrazyflie() for uri in uris}
    swarm.active_uris = set(uris)
    swarm._commander_levels = dict.fromkeys(uris)
    swarm._loop = asyncio.new_event_loop()
    swarm._loop_thread = None
    return swarm


def test_run_schedules_threadsafe_when_loop_is_already_running():
    swarm = make_swarm([])
    swarm._loop_thread = None

    async def command() -> str:
        return "stopped"

    thread = threading.Thread(target=swarm._loop.run_forever)
    thread.start()

    try:
        assert swarm._run(command()) == "stopped"
    finally:
        swarm._loop.call_soon_threadsafe(swarm._loop.stop)
        thread.join()
        swarm._loop.close()


def test_setpoint_sends_once_and_returns():
    uris = ["radio://0/80/2M/E7E7E7E701", "radio://0/80/2M/E7E7E7E702"]
    swarm = make_swarm(uris)
    target = {uris[0]: [1.0, 2.0, 3.0, 4.0], uris[1]: [5.0, 6.0, 7.0, 8.0]}

    try:
        swarm.setpoint(target)
    finally:
        swarm._loop.close()

    for uri in uris:
        cf = swarm.cfs[uri]
        assert cf.fake_commander.setpoints == [tuple(target[uri])]
        assert cf.fake_param.values == [("commander.enHighLevel", 0)]


def test_estimator_updater_copies_batch_and_skips_inactive_drones():
    uris = [
        "radio://0/80/2M/E7E7E7E701",
        "radio://0/80/2M/E7E7E7E702",
        "radio://0/80/2M/E7E7E7E703",
    ]
    swarm = make_swarm(uris)
    swarm.active_uris.remove(uris[2])
    swarm.lighthouse = False
    swarm.update_freq = 1_000
    stop_event = threading.Event()
    connector = FakeROSConnector(
        positions={"cf01": [1.0, 2.0, 3.0], "cf02": [4.0, 5.0, 6.0], "cf03": [7.0, 8.0, 9.0]},
        quaternions={
            "cf01": [0.0, 0.0, 0.0, 1.0],
            "cf02": [0.1, 0.2, 0.3, 0.9],
            "cf03": [0.4, 0.5, 0.6, 0.7],
        },
        stop_event=stop_event,
    )
    swarm.ros_connector = connector
    try:
        asyncio.run(swarm._update_estimators(stop_event))
    finally:
        swarm._loop.close()

    assert swarm.cfs[uris[0]].fake_localization.fake_external_pose.sent == [
        (connector.positions["cf01"], connector.quaternions["cf01"])
    ]
    assert swarm.cfs[uris[1]].fake_localization.fake_external_pose.sent == [
        (connector.positions["cf02"], connector.quaternions["cf02"])
    ]
    assert swarm.cfs[uris[2]].fake_localization.fake_external_pose.sent == []
    assert connector.pos_reads == 1
    assert connector.quat_reads == 1


def test_disconnected_drone_is_warned_and_deactivated(capsys: pytest.CaptureFixture[str]):
    uri = "radio://0/80/2M/E7E7E7E701"
    swarm = make_swarm([uri])
    swarm._commander_levels[uri] = "low"

    async def fail_update(_uri: str) -> None:
        raise DisconnectedError("link lost")

    try:
        asyncio.run(swarm._parallel_by_uri("Updating estimators", [uri], fail_update))
    finally:
        swarm._loop.close()

    assert uri not in swarm.active_uris
    assert swarm._commander_levels[uri] is None
    assert f"{uri} disconnected or unreachable" in capsys.readouterr().err


def test_estimator_updater_lifecycle():
    swarm = make_swarm([])
    swarm.lighthouse = False
    swarm.update_freq = 1_000
    swarm._estimator_stop_event = None
    swarm._estimator_future = None
    swarm._closed = False
    swarm.ros_connector = None

    async def update_until_stopped(stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            await asyncio.sleep(0.001)

    swarm._update_estimators = update_until_stopped
    swarm._start_estimator_updater()

    try:
        assert swarm._loop_thread is not None
        assert swarm._loop_thread.is_alive()
        swarm.close()
        assert swarm._estimator_future is None
        assert swarm._estimator_stop_event is None
    finally:
        if not swarm._loop.is_closed():
            swarm._stop_loop_thread()
            swarm._loop.close()

    assert swarm._loop_thread is None

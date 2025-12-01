"""A simple class to create and deploy a drone swarm."""

from __future__ import annotations

import colorsys
import logging
import os
import struct
import time
from typing import TYPE_CHECKING

import cflib.crtp
import numpy as np
import rclpy
from cflib.crazyflie import Crazyflie, Localization
from cflib.crazyflie.swarm import Swarm
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils.power_switch import PowerSwitch
from drone_estimators.ros_nodes.ros2_connector import ROSConnector

os.environ["SCIPY_ARRAY_API"] = "1"
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from typing import Literal

    from cflib.crayzflie.synccrayzflie import SyncCrazyflie
    from numpy.typing import NDArray as Array

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DroneSwarm:
    """TODO."""

    def __init__(
        self,
        drones: list[dict[str, int]],
        ctrl_freq: float = 50,
        update_freq: float = 10,
        col_freq: float = 1,
    ):
        """TODO.

        Args:
            drones: list of drones with their respective channel and ID
            ctrl_freq: Control frequency (Hz). Defaults to 50.
            update_freq: Frequency (Hz) of position updates sent to the drone. Defaults to 10.
            col_freq: Maximum frequency (Hz) of color updates. Defaults to 1
        """
        self.drones = drones
        self.ctrl_freq = ctrl_freq
        self.update_freq = update_freq
        self.col_freq = col_freq

        self.ros_connector = ROSConnector(
            tf_names=[f"cf{drone['id']}" for drone in drones], timeout=10.0
        )
        self.uris = build_uris(drones)
        cflib.crtp.init_drivers()
        for uri in self.uris:
            PowerSwitch(uri).stm_power_cycle()
        self.swarm = Swarm(self.uris)
        self.swarm.connect()
        ...

    def get_obs(self, uri: str) -> dict[str, Array]:
        drone_name = f"cf{int(uri[-2:], 16):02d}"
        obs = {
            "pos": self.ros_connector.pos[drone_name],
            "quat": self.ros_connector.quat[drone_name],
            # "is_outdated": ros_connector.is_outdated[drone_name],  # this is only available when using an estimator
            # TODO add is_outdated feature to ros_connector for tfs
            "is_outdated": False,
        }
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        return obs

    def takeoff(self, height: float = 1.5, duration: float = 3.0):
        def _parallel_takeoff(scf: SyncCrazyflie, _reporter):
            # send one position update before taking off
            obs = self.get_obs(scf.cf.link_uri)
            scf.cf.extpos.send_extpose(*obs["pos"], *obs["quat"])

            scf.cf.commander.send_stop_setpoint()
            scf.cf.commander.send_notify_setpoint_stop()
            scf.cf.param.set_value("commander.enHighLevel", 1)
            hlc = scf.cf.high_level_commander
            hlc.takeoff(height, duration)
            # keep the onboard estimators updated to avoid drift
            # TODO this should be done via broadcast to reduce load
            t_start = time.time()
            while time.time() < t_start + duration:
                obs = self.get_obs(scf.cf.link_uri)
                scf.cf.extpos.send_extpose(*obs["pos"], *obs["quat"])
                time.sleep(1 / self.update_freq)
            hlc.stop()

        self.swarm.parallel_safe(_parallel_takeoff)

    def land(self, height: float = 0.0, duration: float = 3.0):
        def _parallel_land(scf: SyncCrazyflie, _reporter):
            scf.cf.commander.send_stop_setpoint()
            scf.cf.commander.send_notify_setpoint_stop()
            scf.cf.param.set_value("commander.enHighLevel", 1)
            hlc = scf.cf.high_level_commander
            hlc.land(height, duration)
            # keep the onboard estimators updated to avoid drift
            # TODO this should be done via broadcast to reduce load
            t_start = time.time()
            while time.time() < t_start + duration:
                obs = self.get_obs(scf.cf.link_uri)
                scf.cf.extpos.send_extpose(*obs["pos"], *obs["quat"])
                time.sleep(1 / self.update_freq)
            hlc.stop()

        self.swarm.parallel_safe(_parallel_land)

    def goto(self, pos: dict[str, dict], duration: float = 3.0):
        """Executes a go to command for all drones by linearily interpolating start to desired pos.

        Args:
            pos: Position references in the form {'uri1': {'pos': Array}, ...}
            duration: Duration of the connection in seconds.
        """

        def _parallel_goto(scf: SyncCrazyflie, _reporter, pos: Array):
            scf.cf.param.set_value("commander.enHighLevel", 0)
            pos_start = np.array(
                [*self.get_obs(scf.cf.link_uri)["pos"], self.get_obs(scf.cf.link_uri)["rpy"][0]]
            )
            pos_goal = np.array(pos)
            ref = interp1d([0.0, duration], [pos_start, pos_goal], axis=0)
            t_start = time.time()
            t_est = -np.inf  # Last est update time, starting negative to force an initial update
            while (t := (time.time() - t_start)) < duration:
                if t - t_est >= 1 / self.update_freq:
                    obs = self.get_obs(scf.cf.link_uri)
                    scf.cf.extpos.send_extpose(*obs["pos"], *obs["quat"])
                    t_est = t
                scf.cf.commander.send_position_setpoint(*ref(t))
                time.sleep(1 / self.ctrl_freq)

        assert len(pos.items()) == len(self.drones), (
            "pos does not contain references for all drones."
        )
        self.swarm.parallel_safe(_parallel_goto, pos)

    def execute_choreography(self, choreography: dict[str, dict]):
        """Executes a choreography with position, orientation, and light commands.

        Args:
            choreography: Reference in the form
                          {'uri1': {'t': Array, 'pos': Array, 'color_top': Array, 'color_bot': Array}, ...}
        """

        def _parallel_execution(
            scf: SyncCrazyflie,
            _reporter,
            t: Array,
            pos: Array,
            color_top: Array | None,
            color_bot: Array | None,
        ):
            t_start = time.time()
            scf.cf.param.set_value("commander.enHighLevel", 0)
            t_est = -np.inf  # Last est update time, starting negative to force an initial update
            t_col_top = -np.inf
            t_col_bot = -np.inf
            last_color_top = np.zeros(4)
            last_color_bot = np.zeros(4)
            while (t_cur := (time.time() - t_start)) < t[-1]:
                # estimation
                if t_cur - t_est >= 1 / self.update_freq:
                    obs = self.get_obs(scf.cf.link_uri)
                    scf.cf.extpos.send_extpose(*obs["pos"], *obs["quat"])
                    t_est = t_cur

                # reference
                i = np.searchsorted(t, t_cur)
                scf.cf.commander.send_position_setpoint(*pos[i])

                # color
                if (
                    color_top is not None
                    and t_cur - t_col_top >= 1 / self.col_freq
                    and np.any(last_color_top != color_top[i])
                ):
                    apply_drone_color(scf.cf, color_top[i], "top")
                    t_col_top, last_color_top = t_cur, color_top[i]
                if (
                    color_bot is not None
                    and t_cur - t_col_bot >= 1 / self.col_freq
                    and np.any(last_color_bot != color_bot[i])
                ):
                    apply_drone_color(scf.cf, color_bot[i], "bot")
                    t_col_bot, last_color_bot = t_cur, color_bot[i]
                time.sleep(1 / self.ctrl_freq)

        assert len(choreography.items()) == len(self.drones), (
            "pos does not contain references for all drones."
        )
        self.swarm.parallel_safe(_parallel_execution, choreography)

    def emergency_stop(self, id: int | None = None):
        """Sends an emergency stop signal to one (id) or all drones (default)."""
        pk = CRTPPacket()
        pk.port = CRTPPort.LOCALIZATION
        pk.channel = Localization.GENERIC_CH
        pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
        if id is None:
            for scf in self.swarm._cfs.values():
                scf.cf.send_packet(pk)
        else:
            raise NotImplementedError("Sending emergency stop to one drone not implemented.")
            self.swarm._cf[f"{id}"].cf.send_packet(pk)  # TODO

    def close(self):
        """Closes the swarm and ROS connection."""
        if self.swarm is not None:
            self.swarm.close_link()
        self.ros_connector.close()


def build_uris(drones: list[dict[str, int]]) -> list[str]:
    """This function builds a list of uris to connect the swarm to.

    Finds all available radios and tries to distribute the load of all drones equally
    onto the radios. If there are enough radios, all drones with the same channel are
    set to use one shared radio. Otherwise, groups of channels are assigned.
    """
    radios = cflib.crtp.scan_interfaces()
    uris = [f"{radios}/{drone['channel']}/E7E7E7E7{drone['id']:02x}" for drone in drones]
    ...
    return uris


def apply_drone_color(
    cf: Crazyflie,
    wrgb: Array = np.array([0, 0, 0, 0]),
    deck: Literal["top", "bot", "both"] = "both",
):
    """Applies the given color to the selected deck of a Crazyflie drone."""
    assert np.all((wrgb >= 0) & (wrgb <= 255)), (
        f"Valid range for wrgb values is [0,255], was {wrgb}"
    )
    w, r, g, b = int(wrgb[0]), int(wrgb[1]), int(wrgb[2]), int(wrgb[3])
    if deck == "top" or deck == "both":
        cf.param.set_value("colorLedTop.wrgb8888", int(f"0x{w:02x}{r:02x}{g:02x}{b:02x}", 16))
    if deck == "bot" or deck == "both":
        cf.param.set_value("colorLedBot.wrgb8888", int(f"0x{w:02x}{r:02x}{g:02x}{b:02x}", 16))


def reset_drone(cf: Crazyflie, obs: dict[str, Array]):
    """Resets a given Crazyflie."""
    apply_drone_settings(cf)
    # Reset Kalman filter values
    cf.param.set_value("kalman.initialX", obs["pos"][0])
    cf.param.set_value("kalman.initialY", obs["pos"][1])
    cf.param.set_value("kalman.initialZ", obs["pos"][2])
    cf.param.set_value("kalman.initialYaw", obs["rpy"][2])
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")


def apply_drone_settings(cf: Crazyflie):
    """Apply firmware settings to the drone.

    Note:
        These settings are also required to make the high-level drone commander work properly.
    """
    # Estimator setting;  1: complementary, 2: kalman -> Manual test: kalman significantly better!
    cf.param.set_value("stabilizer.estimator", 2)
    # enable/disable tumble control. Required 0 for agressive maneuvers
    cf.param.set_value("supervisor.tmblChckEn", 1)
    # Choose controller: 1: PID; 2:Mellinger
    cf.param.set_value("stabilizer.controller", 2)
    # rate: 0, angle: 1
    cf.param.set_value("flightmode.stabModeRoll", 1)
    cf.param.set_value("flightmode.stabModePitch", 1)
    cf.param.set_value("flightmode.stabModeYaw", 1)
    time.sleep(0.1)  # Wait for settings to be applied

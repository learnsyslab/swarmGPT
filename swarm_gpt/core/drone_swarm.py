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
    from scipy.interpolate import BSpline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


default_colors = np.array(  # hard coded rainbow colors, https://www.figma.com/color-wheel/
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.5, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.5, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.5, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.5],
    ]
)
default_colors *= 255


class DroneSwarm:
    """TODO."""

    def __init__(
        self,
        drones: dict[str, dict[str, Array]],
        ctrl_freq: float = 50,
        update_freq: float = 10,
        col_freq: float = 1,
    ):
        """TODO.

        Args:
            drones: dictionary of the drones.toml including id, pos, and uri
            ctrl_freq: Control frequency (Hz). Defaults to 50.
            update_freq: Frequency (Hz) of position updates sent to the drone. Defaults to 10.
            col_freq: Maximum frequency (Hz) of color updates. Defaults to 1
        """
        self.drones = drones
        self.ctrl_freq = ctrl_freq
        self.update_freq = update_freq
        self.col_freq = col_freq

        self.ros_connector = ROSConnector(
            tf_names=[f"cf{int(d['uri'][-2:], 16)}" for d in self.drones.values()], timeout=10.0
        )
        self.uris = [d["uri"] for d in self.drones.values()]
        cflib.crtp.init_drivers()
        for uri in self.uris:
            PowerSwitch(uri).stm_power_cycle()
        time.sleep(2)
        self.swarm = Swarm(self.uris)
        self.swarm.open_links()
        self.reset()
        logger.info("init done")

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
        def _parallel_takeoff(scf: SyncCrazyflie):
            try:
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
            except KeyError as e:
                logger.error(f"Taking off failed for {scf.cf.link_uri}: {e}")

        self.swarm.parallel_safe(_parallel_takeoff)

    def land(self, height: float = 0.0, duration: float = 3.0):
        def _parallel_land(scf: SyncCrazyflie):
            try:
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
            except KeyError as e:
                logger.error(f"Landing failed for {scf.cf.link_uri}: {e}")

        self.swarm.parallel_safe(_parallel_land)

    def goto(self, pos: dict[str, list], duration: float = 3.0):
        """Executes a go to command for all drones by linearily interpolating start to desired pos.

        Args:
            pos: Position+Yaw references in the form {'uri1': [pos], ...}
            duration: Duration of the connection in seconds.
        """

        def _parallel_goto(scf: SyncCrazyflie, pos: Array):
            try:
                scf.cf.param.set_value("commander.enHighLevel", 0)
                pos_start = np.array(
                    [*self.get_obs(scf.cf.link_uri)["pos"], self.get_obs(scf.cf.link_uri)["rpy"][0]]
                )
                pos_goal = np.array(pos)
                ref = interp1d([0.0, duration], [pos_start, pos_goal], axis=0)
                t_start = time.time()
                t_est = (
                    -np.inf
                )  # Last est update time, starting negative to force an initial update
                while (t := (time.time() - t_start)) < duration:
                    if t - t_est >= 1 / self.update_freq:
                        obs = self.get_obs(scf.cf.link_uri)
                        scf.cf.extpos.send_extpose(*obs["pos"], *obs["quat"])
                        t_est = t
                    scf.cf.commander.send_position_setpoint(*ref(t))
                    time.sleep(1 / self.ctrl_freq)
            except KeyError as e:
                logger.error(f"Go to failed for {scf.cf.link_uri}: {e}")

        assert len(pos.items()) == len(self.drones), (
            "pos does not contain references for all drones."
        )
        self.swarm.parallel_safe(_parallel_goto, args_dict=pos)

    def execute_choreography(
        self,
        choreography: dict[str, BSpline],
        t_end: float,
        colors: dict[str, dict[str, Array]] | None = None,
    ):
        """Executes a choreography with position, orientation, and light commands.

        Args:
            choreography: Reference in the form of a 3d spline
            t_end: End time of the choreography
            colors: Cues for colors in the form {'t': Array, 'color_top': Array, 'color_bot': Array, 'mode': Array}
                    If None, colors are set to default values.
        """

        def _parallel_execution(
            scf: SyncCrazyflie, choreography: BSpline, t_end: float, colors: dict[str, Array]
        ):
            try:
                t_start = time.time()
                scf.cf.param.set_value("commander.enHighLevel", 0)
                # Last est update time, starting negative to force an initial update
                t_est = -np.inf
                i_next_col_cmd = 0  # Last time we have applied a color command

                while (t_cur := (time.time() - t_start)) < t_end:
                    # estimator update
                    if t_cur - t_est >= 1 / self.update_freq:
                        obs = self.get_obs(scf.cf.link_uri)
                        scf.cf.extpos.send_extpose(*obs["pos"], *obs["quat"])
                        t_est = t_cur

                    # reference # TODO fix hard coded yaw
                    scf.cf.commander.send_position_setpoint(*choreography(t_cur), 0.0)

                    # color
                    if i_next_col_cmd < len(colors["t"]) and t_cur >= colors["t"][i_next_col_cmd]:
                        apply_drone_color(scf.cf, colors["color_top"][i_next_col_cmd], "top")
                        apply_drone_color(scf.cf, colors["color_bot"][i_next_col_cmd], "bot")
                        scf.cf.param.set_value("ledpat.pattern", colors["mode"][i_next_col_cmd])
                        i_next_col_cmd += 1

                    time.sleep(1 / self.ctrl_freq)
            except KeyError as e:
                logger.error(f"Choreography execution failed for {scf.cf.link_uri}: {e}")

        assert len(choreography.items()) == len(self.drones), (
            "pos does not contain references for all drones."
        )

        # build args_dict
        args_dict = {}
        if colors is None:
            colors = {
                # TODO use default colors
                uri: {
                    "t": np.array([0.0]),
                    "color_top": [default_colors[i % len(default_colors)]],
                    "color_bot": [default_colors[i % len(default_colors)]],
                    "mode": np.array([0.0]),
                }
                for i, uri in enumerate(choreography.keys())
            }
        for uri in choreography.keys():
            args_dict[uri] = [choreography[uri], t_end, colors[uri]]
        self.swarm.parallel_safe(_parallel_execution, args_dict=args_dict)

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

    def reset(self):
        """Resets all drones."""
        obs_dict = {}
        for uri in self.uris:
            obs_dict[uri] = [self.get_obs(uri)]
        self.swarm.parallel_safe(reset_drone, args_dict=obs_dict)
        # for scf in self.swarm._cfs.values():
        #     logger.info(f"Resetting {scf.cf.link_uri}")
        #     reset_drone(scf, self.get_obs(scf.cf.link_uri))

    # def connect(self):
    #     self.swarm.open_links()

    def close(self):
        """Closes the swarm and ROS connection."""
        if self.swarm is not None:
            pk = CRTPPacket()
            pk.port = CRTPPort.LOCALIZATION
            pk.channel = Localization.GENERIC_CH
            pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
            for scf in self.swarm._cfs.values():
                scf.cf.send_packet(pk)
            time.sleep(0.1)
            for scf in self.swarm._cfs.values():
                try:
                    # TODO make parallel safe version
                    scf.cf.param.set_value("led.bitmask", 0)  # turn on all LEDs
                    apply_drone_color(scf.cf, np.zeros(4), "both")
                    scf.cf.param.set_value("ledpat.pattern", 0)
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error while closing drone {scf.cf.link_uri}: {e}")
            time.sleep(0.2)  # Wait for commands to be sent
            self.swarm.close_links()
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


def reset_drone(scf: SyncCrazyflie, obs: dict[str, Array]):
    """Resets a given Crazyflie.

    Note:
        These settings are also required to make the high-level drone commander work properly.
    """
    # Estimator setting;  1: complementary, 2: kalman -> Manual test: kalman significantly better!
    scf.cf.param.set_value("stabilizer.estimator", 2)
    # enable/disable tumble control. Required 0 for agressive maneuvers
    scf.cf.param.set_value("supervisor.tmblChckEn", 1)
    # Choose controller: 1: PID; 2:Mellinger
    scf.cf.param.set_value("stabilizer.controller", 2)
    # rate: 0, angle: 1
    scf.cf.param.set_value("flightmode.stabModeRoll", 1)
    scf.cf.param.set_value("flightmode.stabModePitch", 1)
    scf.cf.param.set_value("flightmode.stabModeYaw", 1)
    scf.cf.param.set_value("led.bitmask", 128)  # turn off all LEDs
    time.sleep(0.1)  # Wait for settings to be applied
    # Reset Kalman filter values
    scf.cf.param.set_value("kalman.initialX", obs["pos"][0])
    scf.cf.param.set_value("kalman.initialY", obs["pos"][1])
    scf.cf.param.set_value("kalman.initialZ", obs["pos"][2])
    scf.cf.param.set_value("kalman.initialYaw", obs["rpy"][2])
    scf.cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)  # Wait for settings to be applied
    scf.cf.param.set_value("kalman.resetEstimation", "0")
    scf.cf.platform.send_arming_request(True)
    time.sleep(0.5)  # Wait for motors to start

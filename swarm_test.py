"""Examplary script to deploy a Crazyflie Swarm with position references."""

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
    from cflib.crayzflie.synccrayzflie import SyncCrazyflie
    from numpy.typing import NDArray as Array

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# region helpers


def get_obs(uri: str) -> dict[str, Array]:
    """Generates observations for a given uri."""
    drone_name = f"cf{int(uri[-2:], 16):02d}"
    obs = {
        "pos": ros_connector.pos[drone_name],
        "quat": ros_connector.quat[drone_name],
        # "vel": ros_connector.vel[drone_name],  # this is only available when using an estimator
        # "ang_vel": ros_connector.ang_vel[drone_name],  # this is only available when using an estimator
        # "is_outdated": ros_connector.is_outdated[drone_name],  # this is only available when using an estimator
        # TODO add is_outdated feature to ros_connector for tfs
    }
    obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
    return obs


def reset_drone(cf: Crazyflie):
    """Resets a given Crazyflie."""
    apply_drone_settings(cf)
    obs = get_obs(cf.link_uri)
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
    time.sleep(0.1)  # TODO: Maybe remove
    # enable/disable tumble control. Required 0 for agressive maneuvers
    cf.param.set_value("supervisor.tmblChckEn", 1)
    # Choose controller: 1: PID; 2:Mellinger
    cf.param.set_value("stabilizer.controller", 2)
    # rate: 0, angle: 1
    cf.param.set_value("flightmode.stabModeRoll", 1)
    cf.param.set_value("flightmode.stabModePitch", 1)
    cf.param.set_value("flightmode.stabModeYaw", 1)
    time.sleep(0.1)  # Wait for settings to be applied


def apply_drone_color(cf: Crazyflie, wrgb: Array):
    """Applies the given color to the top and bottop deck of a crazyflie drone."""
    assert np.all((wrgb >= 0) & (wrgb <= 255)), (
        f"Valid range for wrgb values is [0,255], was {wrgb}"
    )
    w, r, g, b = int(wrgb[0]), int(wrgb[1]), int(wrgb[2]), int(wrgb[3])
    cf.param.set_value("colorLedBot.wrgb8888", int(f"0x{w:02x}{r:02x}{g:02x}{b:02x}", 16))
    cf.param.set_value("colorLedTop.wrgb8888", int(f"0x{w:02x}{r:02x}{g:02x}{b:02x}", 16))


def get_reference(t: float) -> Array:
    """Example function to create position references, to be replaced with csv data or splines."""
    i_closest = np.searchsorted(trajectory["t"], t)

    targets = {}  # x, y, z, yaw
    for i, uri in enumerate(URIS):
        targets[uri] = (*trajectory[f"drone{i}_pos"][i_closest], 0)
    return targets


def hl_takeoff(scf: SyncCrazyflie, height: float = 1.0, duration: float = 3.0):
    """Let's the Crazyflie take off and keeps estimators updated."""
    # keep the onboard estimators updated to avoid drift
    pos_start = np.array([*get_obs(scf.cf.link_uri)["pos"], get_obs(scf.cf.link_uri)["rpy"][0]])
    pos_goal = np.array(get_reference(0)[scf.cf.link_uri])
    ref = interp1d([0.0, duration], [pos_start, pos_goal], axis=0)
    t_start = time.time()
    while time.time() < t_start + duration:
        obs = get_obs(scf.cf.link_uri)
        px, py, pz = obs["pos"]
        qx, qy, qz, qw = obs["quat"]
        scf.cf.extpos.send_extpose(px, py, pz, qx, qy, qz, qw)
        x, y, z, yaw = ref(time.time() - t_start)
        scf.cf.commander.send_position_setpoint(x, y, z, yaw)
        time.sleep(1 / pose_update_freq)


def hl_land(scf: SyncCrazyflie, height: float = 0.0, duration: float = 3.0):
    """Let's the Crazyflie land and keeps estimators updated."""
    hlc = scf.cf.high_level_commander
    hlc.land(height, duration)
    # keep the onboard estimators updated to avoid drift
    # TODO this should be done via broadcast to reduce load
    t_start = time.time()
    while time.time() < t_start + duration:
        obs = get_obs(scf.cf.link_uri)
        px, py, pz = obs["pos"]
        qx, qy, qz, qw = obs["quat"]
        scf.cf.extpos.send_extpose(px, py, pz, qx, qy, qz, qw)
        time.sleep(1 / pose_update_freq)
    hlc.stop()


# region control loop
def swarm_pos_ctrl(swarm: Swarm):
    """Main control loop to deploy choreographies.

    TODO This should maybe be done in parallel_safe, but idk how well that scales to large swarms
    """
    # flight_duration = 8.0  # s
    # hover_duration = 4.0  # s

    t_est = 0.0
    t_start = time.time()

    logger.info("Starting combined control loop...")

    # Main tracking loop
    while (t := (time.time() - t_start)) < trajectory["t"][-1]:
        # --- Main 50Hz position control -----------------------------------
        targets = get_reference(t)
        for uri, scf in swarm._cfs.items():
            x, y, z, yaw = targets[uri]
            scf.cf.commander.send_position_setpoint(x, y, z, yaw)

        # --- External pose updates at lower frequency ---------------------
        if t - t_est >= 1 / pose_update_freq:
            for uri, scf in swarm._cfs.items():
                obs = get_obs(uri)
                px, py, pz = obs["pos"]
                qx, qy, qz, qw = obs["quat"]
                scf.cf.extpos.send_extpose(px, py, pz, qx, qy, qz, qw)
            t_est = t

        # timing
        time.sleep(1 / ctrl_freq)

    # Hover at last pos to make sure drone lands in the correct location
    # targets = get_reference(flight_duration)
    # while time.time() - t_start < flight_duration + hover_duration:
    #     # --- External pose updates at lower frequency ---------------------
    #     for uri, scf in swarm._cfs.items():
    #         x, y, z, yaw = targets[uri]
    #         scf.cf.commander.send_position_setpoint(x, y, z, yaw)
    #         obs = get_obs(uri)
    #         px, py, pz = obs["pos"]
    #         qx, qy, qz, qw = obs["quat"]
    #         scf.cf.extpos.send_extpose(px, py, pz, qx, qy, qz, qw)

    #     # timing
    #     time.sleep(1 / pose_update_freq)

    # Stop position commands
    logger.info("enabling HLC")
    for scf in swarm._cfs.values():
        scf.cf.commander.send_stop_setpoint()
        scf.cf.commander.send_notify_setpoint_stop()  # Tell drone to ignore low level setpoints
        scf.cf.param.set_value("commander.enHighLevel", 1)


# region main
if __name__ == "__main__":
    # Settings
    ctrl_freq = 20  # Hz
    pose_update_freq = 10  # Hz
    URIS = [
        "radio://0/100/2M/E7E7E7E70B",
        "radio://0/100/2M/E7E7E7E70C",
        "radio://0/100/2M/E7E7E7E70D",
        "radio://0/100/2M/E7E7E7E70E",
        "radio://0/100/2M/E7E7E7E70F",
        "radio://0/100/2M/E7E7E7E710",
        "radio://0/100/2M/E7E7E7E711",
        "radio://0/100/2M/E7E7E7E712",
        "radio://0/100/2M/E7E7E7E713",
    ]
    colors = {}
    h_values = np.linspace(0, 1, len(URIS))
    for i, uri in enumerate(URIS):
        rgb = np.array(colorsys.hsv_to_rgb(h_values[i], 1, 1))
        rgb /= np.linalg.norm(rgb)
        rgb *= 255
        colors[uri] = np.array([0, *rgb])
    # colors = {
    #     "radio://0/100/2M/E7E7E7E70B": np.array([0, 255, 0, 0]),
    #     "radio://0/100/2M/E7E7E7E70C": np.array([0, 0, 128, 128]),
    #     "radio://0/100/2M/E7E7E7E70D": np.array([0, 128, 128, 0]),
    #     "radio://0/100/2M/E7E7E7E70E": np.array([0, 128, 0, 128]),
    #     "radio://0/100/2M/E7E7E7E70F": np.array([0, 0, 255, 0]),
    #     "radio://0/100/2M/E7E7E7E710": np.array([0, 128, 64, 64]),
    #     "radio://0/100/2M/E7E7E7E711": np.array([0, 64, 64, 128]),
    #     "radio://0/100/2M/E7E7E7E712": np.array([0, 64, 128, 64]),
    #     "radio://0/100/2M/E7E7E7E713": np.array([0, 0, 0, 255]),
    # }

    # load reference. TODO remove hard coded part
    data = np.loadtxt("trajectory.csv", delimiter=",", skiprows=1)

    trajectory = {
        "t": data[:, 0],
        "drone0_pos": data[:, 1:4],
        "drone1_pos": data[:, 7:10],
        "drone2_pos": data[:, 13:16],
        "drone3_pos": data[:, 19:22],
        "drone4_pos": data[:, 25:28],
        "drone5_pos": data[:, 31:34],
        "drone6_pos": data[:, 37:40],
        "drone7_pos": data[:, 43:46],
        "drone8_pos": data[:, 49:52],
    }

    rclpy.init()
    ros_connector = ROSConnector(
        # estimator_names=[f"cf{int(uri[-2:], 16):02d}" for uri in URIS],  # this is only available when using an estimator
        tf_names=[f"cf{int(uri[-2:], 16):02d}" for uri in URIS],
        timeout=10.0,
    )

    cflib.crtp.init_drivers()

    import usb.core
    import usb.util

    # Crazyradio USB IDs
    VID = 0x1915
    PID = 0x7777

    # Find Crazyradio
    dev = usb.core.find(idVendor=VID, idProduct=PID)
    if dev is None:
        raise RuntimeError("Crazyradio dongle not found.")

    # 0x10 = ACK_ENABLE command in Crazyradio firmware
    # wValue = 0 disables ACK
    dev.ctrl_transfer(
        bmRequestType=0x40,  # Vendor OUT
        bRequest=0x10,  # ACK_ENABLE
        wValue=0,  # 0 = disable
        wIndex=0,
        data_or_wLength=None,
    )

    logger.info("Restarting all Crazyflies...")
    for uri in URIS:
        PowerSwitch(uri).stm_power_cycle()
    time.sleep(0.5)  # Wait for all drones to complete the reboot

    # TODO to not be reliant on sending pos updates (see functions above), it would be helpful
    # if we could simply give a get_obs() or get_obs(uri) function which is then internally
    # called and all estimators are updated in a batched/broadcast manner.
    logger.info("Connecting to Crazyflies...")
    with Swarm(URIS) as swarm:
        try:
            logger.info("Setting up Crazyflie settings...")
            for scf in swarm._cfs.values():
                reset_drone(scf.cf)

            # TODO do not wait for ack

            time.sleep(1.0)

            logger.info("Arming all Crazyflies...")
            for scf in swarm._cfs.values():
                scf.cf.platform.send_arming_request(True)
                time.sleep(0.1)

            time.sleep(2.0)

            # Smooth HLC takeoff
            logger.info("Taking off...")
            swarm.parallel_safe(hl_takeoff)
            # TODO drones drift during takeoff

            for scf in swarm._cfs.values():
                apply_drone_color(scf.cf, colors[scf.cf.link_uri])

            # Run main control loop + slow estimator updates
            logger.info("Starting choreography...")
            swarm_pos_ctrl(swarm)

            # Turn lights off
            for scf in swarm._cfs.values():
                apply_drone_color(scf.cf, colors[scf.cf.link_uri] * 0)

            # Smooth HLC landing
            logger.info("Landing...")
            swarm.parallel_safe(hl_land)
            # TODO drones drift during landing (and are not at the correct pos)

        finally:
            try:
                # Emergency stop when ctrl+C
                pk = CRTPPacket()
                pk.port = CRTPPort.LOCALIZATION
                pk.channel = Localization.GENERIC_CH
                pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
                for scf in swarm._cfs.values():
                    scf.cf.send_packet(pk)
                for scf in swarm._cfs.values():
                    scf.cf.close_link()
            finally:
                # Close all ROS connections
                ros_connector.close()

"""Examplary script to deploy a Crazyflie Swarm with position references."""

from __future__ import annotations

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
from cflib.crtp.radiodriver import RadioDriver
from cflib.utils.power_switch import PowerSwitch
from drone_estimators.ros_nodes.ros2_connector import ROSConnector

os.environ["SCIPY_ARRAY_API"] = "1"
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

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


def get_reference(t: float, t_total: float = 10, n_turns: int = 1) -> Array:
    """Example function to create position references, to be replaced with csv data or splines."""
    R = 1.5
    z = 1.0
    theta = 2 * np.pi * (n_turns * t / t_total)

    targets = {}  # x, y, z, yaw
    total_URIs = len(URIS)
    for i, uri in enumerate(URIS):
        targets[uri] = (
            R * np.cos(theta + 2 * i / total_URIs * np.pi),
            R * np.sin(theta + 2 * i / total_URIs * np.pi),
            z,
            0,
        )
    targets["theta"] = theta
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
    flight_duration = 8.0  # s
    hover_duration = 4.0  # s
    n_turns = 2

    t_est = 0.0
    t_start = time.time()

    logger.info("Starting combined control loop...")

    # Main tracking loop
    while (t := (time.time() - t_start)) < flight_duration:
        # --- Main 50Hz position control -----------------------------------
        targets = get_reference(t, flight_duration, n_turns=n_turns)
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
    targets = get_reference(flight_duration, flight_duration, n_turns=n_turns)
    while time.time() - t_start < flight_duration + hover_duration:
        # --- External pose updates at lower frequency ---------------------
        for uri, scf in swarm._cfs.items():
            x, y, z, yaw = targets[uri]
            scf.cf.commander.send_position_setpoint(x, y, z, yaw)
            obs = get_obs(uri)
            px, py, pz = obs["pos"]
            qx, qy, qz, qw = obs["quat"]
            scf.cf.extpos.send_extpose(px, py, pz, qx, qy, qz, qw)

        # timing
        time.sleep(1 / pose_update_freq)

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
    time.sleep(2)  # Wait for all drones to complete the reboot

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

            time.sleep(2.0)

            logger.info("Arming all Crazyflies...")
            for scf in swarm._cfs.values():
                scf.cf.platform.send_arming_request(True)
                time.sleep(0.5)

            time.sleep(2.0)

            # Smooth HLC takeoff
            logger.info("Taking off...")
            swarm.parallel_safe(hl_takeoff)
            # TODO drones drift during takeoff

            # Run main control loop + slow estimator updates
            logger.info("Starting choreography...")
            swarm_pos_ctrl(swarm)

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

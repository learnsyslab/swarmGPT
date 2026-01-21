import os
import time

import cflib.crtp
import numpy as np
import rclpy
from cflib.crazyflie import Crazyflie
from drone_estimators.ros_nodes.ros2_connector import ROSConnector

os.environ["SCIPY_ARRAY_API"] = "1"
from scipy.spatial.transform import Rotation as R

URI = "radio://0/10/2M/E7E7E7E70B"


def connected(link_uri):
    print(f"Connected to {link_uri}")


def disconnected(link_uri):
    print(f"Disconnected from {link_uri}")


def connection_failed(link_uri, msg):
    print(f"Connection failed: {msg}")


def connection_lost(link_uri, msg):
    print(f"Connection lost: {msg}")


def get_obs(uri: str):
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


if __name__ == "__main__":
    rclpy.init()
    ros_connector = ROSConnector(tf_names=[f"cf{int(URI[-2:], 16)}"], timeout=10.0)

    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    # Create the Crazyflie instance
    cf = Crazyflie(rw_cache="./cache")

    # Register callbacks
    cf.connected.add_callback(connected)
    cf.disconnected.add_callback(disconnected)
    cf.connection_failed.add_callback(connection_failed)
    cf.connection_lost.add_callback(connection_lost)
    cf.console.receivedChar.add_callback(
        lambda msg: print(f"drone: {msg.strip().replace('\n', '').replace('\r', '')}")
    )

    # Connect (non-blocking)
    cf.open_link(URI)

    # Wait until parameters are downloaded
    print("Waiting for parameter download...")
    while not cf.param.is_updated:
        time.sleep(0.1)
    cf.param.set_value("stabilizer.estimator", 2)  # Set to EKF

    print("Setting LED color…")

    try:
        cf.param.set_value("ledpat.pattern", 0)
        for i in range(0, 255, 10):
            cf.param.set_value("colorLedBot.wrgb8888", int(f"0x000000{255 - i:02x}", 16))
            cf.param.set_value("colorLedTop.wrgb8888", int(f"0x00{i:02x}0000", 16))
            time.sleep(0.05)
        for i in range(0, 255, 10):
            cf.param.set_value("colorLedBot.wrgb8888", int(f"0x000000{i:02x}", 16))
            cf.param.set_value("colorLedTop.wrgb8888", int(f"0x00{255 - i:02x}0000", 16))
            time.sleep(0.05)
        cf.param.set_value("ledpat.pattern", 5)
        time.sleep(2.0)
        while True:
            cf.param.set_value("ledpat.pattern", 2)
            obs = get_obs(URI)
            cf.extpos.send_extpose(*obs["pos"], *obs["quat"])
            time.sleep(0.1)

    finally:
        # cf.param.set_value("colorLedBot.wrgb8888", int("0xFF000000", 16))  # solid color mode
        # cf.param.set_value("colorLedTop.wrgb8888", int("0xFF000000", 16))  # solid color mode

        # time.sleep(2.0)

        # cf.param.set_value("colorLedBot.wrgb8888", int("0xFFFFFFFF", 16))  # solid color mode
        # cf.param.set_value("colorLedTop.wrgb8888", int("0xFFFFFFFF", 16))  # solid color mode

        # time.sleep(2.0)

        cf.param.set_value("colorLedBot.wrgb8888", int("0x00000000", 16))  # solid color mode
        cf.param.set_value("colorLedTop.wrgb8888", int("0x00000000", 16))  # solid color mode
        cf.param.set_value("ledpat.pattern", 0)
        time.sleep(0.1)  # make sure packages get sent out

        # Close link
        cf.close_link()

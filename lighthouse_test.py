import os
import time

import cflib.crtp
import numpy as np
import rclpy
from cflib.crazyflie import Crazyflie

URI = "radio://0/80/2M/E7E7E7E70A"


def connected(link_uri):
    print(f"Connected to {link_uri}")


def disconnected(link_uri):
    print(f"Disconnected from {link_uri}")


def connection_failed(link_uri, msg):
    print(f"Connection failed: {msg}")


def connection_lost(link_uri, msg):
    print(f"Connection lost: {msg}")


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
    cf.platform.send_arming_request(True)
    time.sleep(0.8)  # Wait for motors to start


def main():
    """Main control loop to deploy choreographies.

    TODO This should maybe be done in parallel_safe, but idk how well that scales to large swarms
    """
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

    apply_drone_settings(cf)

    try:
        cf.high_level_commander.takeoff(1.0, 2.0)
        time.sleep(2.0)
        cf.high_level_commander.land(0.0, 2.0)
        time.sleep(2.0)

    finally:
        cf.commander.send_stop_setpoint()
        cf.commander.send_notify_setpoint_stop()  # Tell drone to ignore low level setpoints
        cf.close_link()


if __name__ == "__main__":
    main()

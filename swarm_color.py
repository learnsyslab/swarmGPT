import time
import cflib.crtp
from cflib.crazyflie import Crazyflie

URI = "radio://0/100/2M/E7E7E7E714"


def connected(link_uri):
    print(f"Connected to {link_uri}")


def disconnected(link_uri):
    print(f"Disconnected from {link_uri}")


def connection_failed(link_uri, msg):
    print(f"Connection failed: {msg}")


def connection_lost(link_uri, msg):
    print(f"Connection lost: {msg}")


# Initialize the low-level drivers
cflib.crtp.init_drivers()

# Create the Crazyflie instance
cf = Crazyflie(rw_cache="./cache")

# Register callbacks
cf.connected.add_callback(connected)
cf.disconnected.add_callback(disconnected)
cf.connection_failed.add_callback(connection_failed)
cf.connection_lost.add_callback(connection_lost)

# Connect (non-blocking)
cf.open_link(URI)

# Wait until parameters are downloaded
print("Waiting for parameter download...")
while not cf.param.is_updated:
    time.sleep(0.1)

print("Setting LED color…")

try:
    while True:
        for i in range(0, 255, 5):
            cf.param.set_value("colorLedBot.wrgb8888", int(f"0x000000{255 - i:02x}", 16))
            cf.param.set_value("colorLedTop.wrgb8888", int(f"0x00{i:02x}0000", 16))
            time.sleep(0.05)
        for i in range(0, 255, 5):
            cf.param.set_value("colorLedBot.wrgb8888", int(f"0x000000{i:02x}", 16))
            cf.param.set_value("colorLedTop.wrgb8888", int(f"0x00{255 - i:02x}0000", 16))
            time.sleep(0.05)

finally:
    cf.param.set_value("colorLedBot.wrgb8888", int("0xFF000000", 16))  # solid color mode
    cf.param.set_value("colorLedTop.wrgb8888", int("0xFF000000", 16))  # solid color mode

    time.sleep(2.0)

    cf.param.set_value("colorLedBot.wrgb8888", int("0xFFFFFFFF", 16))  # solid color mode
    cf.param.set_value("colorLedTop.wrgb8888", int("0xFFFFFFFF", 16))  # solid color mode

    time.sleep(2.0)

    cf.param.set_value("colorLedBot.wrgb8888", int("0x00000000", 16))  # solid color mode
    cf.param.set_value("colorLedTop.wrgb8888", int("0x00000000", 16))  # solid color mode

    time.sleep(0.1)  # make sure packages get sent out

    # Close link
    cf.close_link()

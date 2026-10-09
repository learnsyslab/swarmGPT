import asyncio
from cflib2 import Crazyflie, LinkContext

async def main():
    context = LinkContext()

    uris = await context.scan(address=list(bytes.fromhex("E7E7E7E70F")))
    print(f"Found: {uris}")

    cf = await Crazyflie.connect_from_uri(context, "radio://0/10/2M/E7E7E7E70F")
    param = cf.param()
    value = await param.get("pm.lowVoltage")
    print(f"Low voltage threshold: {value}V")
    await cf.disconnect()

asyncio.run(main())
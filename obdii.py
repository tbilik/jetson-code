import obd
from obd import OBDStatus
import time
import math

obd.logger.setLevel(obd.logging.DEBUG)

def retrieveData(pid):
    return connection.query(obd.commands[pid]).value

# connect to ELM327 OBD-II interface chip
while True:
    try:
        connection = obd.Async(fast=False,timeout=30)
        if connection.status() == OBDStatus.CAR_CONNECTED:
            break
    except:
        # this code runs if the connection fails in some way
        print("Connection failed. Trying again ...") # this is for the log file

connection.watch(obd.commands["SPEED"])
# start connection of interface chip with vehicle
connection.start()
# approximately 2 seconds needed to stabilize connection
time.sleep(2)

while True:
    while connection.status() != OBDStatus.CAR_CONNECTED:
        with open("/home/tbilik/display_fifo","w") as fp:
            fp.write("C")
        time.sleep(1)
    with open("/home/tbilik/display_fifo","w") as fp:
        speed = int(retrieveData("SPEED").to("mph").magnitude)
        fp.write("A%d\n" % (speed,))
    time.sleep(1)

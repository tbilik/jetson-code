import time
import board
import busio
import os
import atexit
import select as select
from adafruit_ht16k33 import segments
 
# Create the I2C interface.
i2c = busio.I2C(board.SCL, board.SDA)
 
# Create the LED segment class.
# This creates a 7 segment 4 character display:
display = segments.Seg7x4(i2c)
 
# Clear the display.
display.fill(0)
 
# Can just print a number
display.print(":")
 
# Set the first character to '1':
display[0] = '0'
# Set the second character to '2':
display[1] = '0'
# Set the third character to 'A':
display[2] = '0'
# Set the forth character to 'B':
display[3] = '0'
 
numbers = [0.0, 1.0, -1.0, 0.55, -0.55, 10.23, -10.2, 100.5]

with open("display_fifo") as fifo:
    while True:
        select.select([fifo],[],[fifo])
        data = fifo.read().rstrip()
        if data == "":
            continue
        if data[0] == "A":
            if len(data) == 2:
                display[0] = "0"
                display[1] = data[1]
            elif len(data) == 3:
                display[0] = data[1]
                display[1] = data[2]
            else:
                display[0] = "9"
                display[1] = "9"
        if data[0] == "B":
            if data[1:] == "ten":
                display[2] = "1"
                display[3] = "0"

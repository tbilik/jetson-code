import time
import board
import busio
import os
import atexit
import select as select
from adafruit_ht16k33 import segments
import RPi.GPIO as GPIO

# determines whether the buzzer should be used
alarm = True
obdOn = False

# define pin values
buttonPin = 19
buzzerPin = 32
speedLimit = 0
currentSpeed = 0

# set pin states
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(buttonPin, GPIO.IN)
GPIO.setup(buzzerPin, GPIO.OUT)
buzzer = GPIO.PWM(buzzerPin, 2000)

# Create the I2C interface.
i2c = busio.I2C(board.SCL, board.SDA)
 
# Create the LED segment class.
# This creates a 7 segment 4 character display:
display = segments.Seg7x4(i2c)

def toggle_alarm(channel):
    global alarm
    global buzzer
    global display
    global currentSpeed
    if currentSpeed >= 90:
        display[0] = "9"
    else:
        display[0] = str(currentSpeed // 10) 
    buzzer.stop()
    alarm = not alarm
    if alarm:
        display[0] = "."

GPIO.add_event_detect(buttonPin, GPIO.RISING, callback=toggle_alarm, bouncetime=200)

# Clear the display.
display.fill(0)
 
# Can just print a number
display.print(":")
 
# Set the first character to '1':
display[0] = '0'
display[0] = '.'
# Set the second character to '2':
display[1] = '0'
# Set the third character to 'A':
display[2] = '0'
# Set the forth character to 'B':
display[3] = '0'
 
# numbers = [0.0, 1.0, -1.0, 0.55, -0.55, 10.23, -10.2, 100.5]

with open("display_fifo") as fifo:
    while True:
        select.select([fifo],[],[fifo])
        data = fifo.read().rstrip()
        if data == "":
            continue
        if data[0] == "A":
            obdOn = True
            if len(data) == 2:
                display[0] = "0"
                display[1] = data[1]
            elif len(data) == 3:
                display[0] = data[1]
                display[1] = data[2]
            else:
                display[0] = "9"
                display[1] = "9"
            display[1] = "."
            if alarm:
                display[0] = "."
            currentSpeed = int(data[1:])
        if data[0] == "C":
            if currentSpeed >= 99:
                display[1] = "9"
            else:
                display[1] = str(currentSpeed % 10)
            obdOn = False
        if data[0] == "B":
            if data[1:] == "ten":
                display[2] = "1"
                display[3] = "0"
                speedLimit = 10
            if data[1:] == "fifteen":
                display[2] = "1"
                display[3] = "5"
                speedLimit = 15
            if data[1:] == "twenty":
                display[2] = "2"
                display[3] = "0"
                speedLimit = 20
            if data[1:] == "twentyfive":
                display[2] = "2"
                display[3] = "5"
                speedLimit = 25
            if data[1:] == "thirty":
                display[2] = "3"
                display[3] = "0"
                speedLimit = 30
            if data[1:] == "thirtyfive":
                display[2] = "3"
                display[3] = "5"
                speedLimit = 35
            if data[1:] == "forty":
                display[2] = "4"
                display[3] = "0"
                speedLimit = 40
            if data[1:] == "fortyfive":
                display[2] = "4"
                display[3] = "5"
                speedLimit = 45
            if data[1:] == "fifty":
                display[2] = "5"
                display[3] = "0"
                speedLimit = 50
            if data[1:] == "fiftyfive":
                display[2] = "5"
                display[3] = "5"
                speedLimit = 55
            if data[1:] == "sixty":
                display[2] = "6"
                display[3] = "0"
                speedLimit = 60
            if data[1:] == "sixtyfive":
                display[2] = "6"
                display[3] = "5"
                speedLimit = 65
            if data[1:] == "seventy":
                display[2] = "7"
                display[3] = "0"
                speedLimit = 70
        if alarm and obdOn and speedLimit != 0 and currentSpeed - speedLimit >= 10:
            buzzer.start(90)
        else:
            buzzer.stop()

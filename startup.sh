#!/bin/bash

rfcomm bind 0 00:1D:A5:00:67:FE 1 &
cd /home/tbilik/
python3 jetson-code/display_driver.py &
python3 jetson-code/obdii.py &
python3 jetson-code/sign-detect.py camera &

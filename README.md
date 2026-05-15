# Muraqib
a surveillance system that doesnt cost u ur privacy.
AI Safety and Privacy First Surveillance Prototype


WHAT IT IS

Muraqib is a small surveillance system that runs on a Raspberry Pi 5. It
watches a live webcam feed and uses AI to detect people and bags. Faces
are automatically blurred so identity is never stored. When a bag is
detected, a simulated thermal scan activates and shows a weapon-like
object inside it. Everything is shown on a web dashboard that opens in
any browser.

The system was built as a demo to explore how surveillance technology
can be designed with privacy as the default, not an afterthought.


FILES IN THIS FOLDER

muraqib.py        The main program. Runs on the Raspberry Pi.
dashboard.html    The dashboard. Opens in a browser on any device.
README.txt        This file.


REQUIREMENTS

Raspberry Pi 5 with Python 3 installed
A USB webcam
Wi-Fi connection
yolo11s.pt model file in the same folder as muraqib.py

Python packages needed:
  ultralytics
  opencv-python
  numpy


HOW TO RUN

1. Copy both files to the Raspberry Pi in the same folder.

2. Make sure yolo11s.pt is in the folder too.

3. Open a terminal and run:
       python muraqib.py

4. The program will print something like:
       Stream: http://192.168.1.105:5000/stream

5. Open dashboard.html on any device on the same WiFi network.

6. Type the Pi IP address into the box at the bottom right of the
   dashboard and click CONNECT.

7. To stop the system, press Ctrl+C in the terminal.


WHAT THE DASHBOARD SHOWS

Live camera feed with face blur applied
Status bar showing CLEAR, BAG DETECTED, or THREAT
Person cards with detected clothing colors
Show Whereabouts button to see each person's movement
Event timeline with timestamps
Weapon zoom panel during a thermal scan
Zone minimap showing where each person is
Reset Alert and Export Log buttons


NOTES

The weapon detection is a visual simulation only. The system draws a
gun shaped overlay inside any detected bag to show what a real threat
detection system might look like. No actual weapon classification is
done.

Faces are blurred using OpenCV face detection. No facial recognition
is performed and no images are saved.

Person IDs reset every time the program is restarted.

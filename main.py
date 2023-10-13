# Libraries
import mediapipe as mp  # Import MediaPipe for Hand Detection
import numpy as np  # Import NumPy for Array Manipulation
import cv2 as cv  # Import OpenCV for Image Manipulation
import sys  # Import System for Error Handling

# Other Python Files
from detect import hand_detection, coin_detection  # Import Hand Detection Function
from macros import fprint, waitUser  # Import Print Function


# Define Absolute Path to Model
modelPath = sys.path[0] + "hand_landmarker.task"

# Open & Setup Camera
fprint("M", "Opening Camera...")
cam = cv.VideoCapture(0)
if not cam.isOpened():
    fprint("E", "Cannot open camera")
    exit()
# Downsizing Camera (For Performance)
cam.set(cv.CAP_PROP_FPS, 60)  # Set FPS to 60
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Set Width to 640
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Set Height to 480
fprint("M", "Camera Opened!")

# Wait for User to Press Space
fprint("C", "<-- Press Space to Capture Background -->")
waitUser(" ", cam)
fprint("M", "Capturing Background...")

# Save 1st Frame as Background
ret, frame = cam.read()
if not ret:  # if frame is read correctly ret is True
    # eprint("Can't receive frame (stream end?). Exiting ...")
    exit()
cv.imwrite("background.png", frame)
fprint("M", "Background Captured!")

# Wait for User to Press Space
fprint("C", "<-- Press Space to Capture Coin Position -->")
waitUser(" ", cam)
fprint("M", "Capturing Coin Position...")

# Save 2nd Frame with Coin Position
ret, frame = cam.read()
if not ret:  # if frame is read correctly ret is True
    # eprint("Can't receive frame (stream end?). Exiting ...")
    exit()
coinPos = coin_detection(frame)
cv.imwrite("coin.png", frame)
fprint("M", "Coin Position Captured!")


# Handle While Camera is Open
while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    if not ret:  # if frame is read correctly ret is True
        fprint("E", "Can't receive frame (stream end?). Exiting ...")
        break

    # Detect Hand
    hands, frame = hand_detection(frame, draw=True)

    # Display the resulting frame
    cv.imshow("frame", frame)

    # Handle Exit
    if cv.waitKey(1) == ord("q"):
        break

# Release & Destroy Camera
cam.release()  # Release camera
cv.destroyAllWindows()  # Destroy all windows

# Libraries
import cv2 as cv  # Import OpenCV for Image Manipulation
import sys  # Import System for Error Handling

# Other Python Files
from detect import (
    hand_detection,
    coin_detection,
    isTouching,
)  # Import Detection Function
from macros import fprint, waitUser  # Import Print Function


mode = "SHOW"
hover = False
touch = 0

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
    fprint("E", "Can't receive frame (stream end?). Exiting ...")
    exit()
cv.imwrite("background.png", frame)
fprint("M", "Background Captured!")

# Wait for User to Press Space
fprint("C", "<-- Press Space to Capture Coin Position -->")

# Save 2nd Frame with Coin Position
while True:
    ret, frame = cam.read()
    if not ret:  # if frame is read correctly ret is True
        fprint("E", "Can't receive frame (stream end?). Exiting ...")
        exit()
    coinPos, frame = coin_detection(frame, draw=True)  # Detect Coin
    cv.imshow("frame", frame)  # Display frame
    if cv.waitKey(1) == ord(" "):  # Handle Exit
        break

fprint("M", "Capturing Coin Position...")
cv.imwrite("coin.png", frame)
fprint("M", "Coin Position Captured!")


# Handle While Camera is Open
while True:
    ret, frame = cam.read()
    if not ret:  # if frame is read correctly ret is True
        fprint("E", "Can't receive frame (stream end?). Exiting ...")
        break

    # Detect Hand
    hands, frame = hand_detection(frame)

    if isTouching(hands, coinPos) and not hover:
        print("Touched!")
        hover = True
        touch += 1
    elif not isTouching(hands, coinPos) and hover:
        print("Leaved!")
        hover = False

    mode = "SHOW" if touch % 2 == 0 else "HIDE"

    # Cover the Coin base on the Coin Position with the Background
    background = cv.imread("background.png")
    deviation = 20
    if mode == "HIDE":
        for circle in coinPos:
            x, y, r = circle[0], circle[1], circle[2] + deviation
            stX, stY, edX, edY = x - r, y - r, x + r, y + r
            frame[stY:edY, stX:edX] = background[stY:edY, stX:edX]

    # Display the resulting frame
    cv.imshow("frame", frame)

    # Manual Trigger Mode
    if cv.waitKey(1) == ord(" "):
        print("Triggered!")
        touch += 1

    # Handle Exit
    if cv.waitKey(1) == ord("q"):
        break

# Release & Destroy Camera
cam.release()  # Release camera
cv.destroyAllWindows()  # Destroy all windows

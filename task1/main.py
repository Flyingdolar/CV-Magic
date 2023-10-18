# Libraries
import cv2 as cv  # Import OpenCV for Image Manipulation
import sys  # Import System for Error Handling

# Other Python Files
from detect import (
    hand_detection,
    coin_detection,
    check_touch,
    hide_coin,
)  # Import Detection Function
from macros import fprint, waitUser  # Import Print Function


# Define Absolute Path to Model
modelPath = sys.path[0] + "hand_landmarker.task"

# Open & Setup Camera
fprint("M", "Opening Camera...")
# Open a Video File instead of Camera
cam = cv.VideoCapture("coin1_big_1.mkv")
if not cam.isOpened():
    fprint("E", "Cannot open camera")
    exit()
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
    coinsPos, frame = coin_detection(frame, draw=True)  # Detect Coin
    # Create a Coins state to save individual coin state: touch: {number of touch}, hover: {is hover}
    coins = (
        coinsPos is not None
        and [
            {"times": 0, "hover": False, "x": cPos[0], "y": cPos[1], "r": cPos[2]}
            for cPos in coinsPos
        ]
        or None
    )
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
    hands, frame = hand_detection(frame, draw=True)

    # Do nothing if there is no coins
    if coins is None:
        cv.imshow("frame", frame)
        continue

    # Check if hand is touching the coin
    if hands is not None:
        touchList = check_touch(hands, coins)
        if touchList is not None:
            for coin in coins:
                if coin in touchList and coin.hover is False:
                    coin.times += 1
                    coin.hover = True
                elif coin not in touchList and coin.hover is True:
                    coin.hover = False

    # Hide the coin if it is touched odd times
    for coin in coins:
        if coin.times % 2 == 1:
            frame = hide_coin(frame, hands, coin)

    # Display the resulting frame
    cv.imshow("frame", frame)

    # User Triggered Exit or Toggle Mode
    usrCmd = cv.waitKey(1)
    if usrCmd == ord("q") or usrCmd == 27:
        break


# Release & Destroy Camera
cam.release()  # Release camera
cv.destroyAllWindows()  # Destroy all windows

import mediapipe as mp
import cv2 as cv
import sys


# Define Print Error
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# Define Absolute Path to Model
modelPath = sys.path[0] + "hand_landmarker.task"

# Open Camera
cam = cv.VideoCapture(0)
if not cam.isOpened():
    eprint("Cannot open camera")
    exit()


# Handle While Camera is Open
while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    if not ret:  # if frame is read correctly ret is True
        eprint("Can't receive frame (stream end?). Exiting ...")
        break

    # TODO: Do Magic Trick Here
    # ...

    # Display the resulting frame
    cv.imshow("frame", frame)

    # Handle Exit
    if cv.waitKey(1) == ord("q"):
        break

# Release & Destroy Camera
cam.release()  # Release camera
cv.destroyAllWindows()  # Destroy all windows

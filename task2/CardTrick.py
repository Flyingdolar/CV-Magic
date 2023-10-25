import cv2 as cv  # Import OpenCV for Image Manipulation
import sys  # Import System for Error Handling
import numpy as np  # Import Numpy for Array Operations

sys.path.append("../")  # Add parent directory to Python Path

from Macros import fprint  # Import Print Function


def apply_card_trick(cam):
    cam.stPrint("Start Card Trick")
    # Change to HSV Color Space
    hsv = cv.cvtColor(cam.fr, cv.COLOR_BGR2HSV)
    # Get white points of the frame
    light = max(hsv[:, :, 2].flatten())
    light = 200 if light < 200 else light
    satur = 30
    # Create a mask for the white points
    mask = cv.inRange(hsv, (0, 0, light - 50), (255, satur, 255))
    # Find the contours of the white points
    contours, _ = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    # Find the biggest contour
    maxArea = 0
    maxContour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > maxArea:
            maxArea = area
            maxContour = contour
    # Remove the holes in the contour
    hull = cv.convexHull(maxContour)
    # Draw the contour
    cv.drawContours(cam.fr, [hull], -1, (0, 255, 0), -1)
    # Get the corners of the contour
    corners = cv.approxPolyDP(hull, 0.1 * cv.arcLength(hull, True), True)
    # Paste the card image on the frame by the corners
    if len(corners) == 4:
        pts1 = np.float32(corners)
        pts2 = np.float32(
            [[0, 0], [0, 480], [640, 480], [640, 0]]
        )
        M = cv.getPerspectiveTransform(pts1, pts2)
        cam.fr = cv.warpPerspective(
            cam.fr, M, (640, 480)
        )

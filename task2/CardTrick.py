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
    lowerb = np.array([0, 0, light - 50])
    upperb = np.array([255, satur, 255])
    mask = cv.inRange(hsv, lowerb, upperb)
    # Find the contours of the white points
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Find the biggest contour
    maxArea = 0
    maxContour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > maxArea:
            maxArea = area
            maxContour = contour
    if maxContour is None:
        return cam
    # Remove the holes in the contour
    hull = cv.convexHull(maxContour)
    # Get the corners of the contour
    corners = cv.approxPolyDP(hull, 0.1 * cv.arcLength(hull, True), True)
    # Draw the corners
    for corner in corners:
        cv.circle(cam.fr, tuple(corner[0]), 5, (0, 0, 255), -1)
    # Paste the card image on the frame by the corners
    if len(corners) == 4:
        card = cv.imread("cards/" + str(cam.card) + ".png")  # Read the card image
        card = cv.resize(card, (cam.fr.shape[1], cam.fr.shape[0]))
        cardH, cardW, _ = card.shape
        # Get the perspective transform matrix
        pts1 = np.float32([[0, 0], [cardW, 0], [cardW, cardH], [0, cardH]])
        # Let the corner sequence be top-left, top-right, bottom-left, bottom-right
        np.sort(corners, axis=0)
        # Draw Four Corners in different colors on the frame
        cv.circle(cam.fr, tuple(corners[0][0]), 5, (255, 0, 0), -1)  # Blue
        cv.circle(cam.fr, tuple(corners[1][0]), 5, (0, 255, 0), -1)  # Green
        cv.circle(cam.fr, tuple(corners[2][0]), 5, (0, 0, 255), -1)  # Red
        cv.circle(cam.fr, tuple(corners[3][0]), 5, (255, 255, 0), -1)  # Cyan
        pts2 = np.float32([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
        Mtx = cv.getPerspectiveTransform(pts1, pts2)
        # Apply the perspective transform matrix
        card = cv.warpPerspective(card, Mtx, (cam.fr.shape[1], cam.fr.shape[0]))
        # Make a mask for the card
        cardMsk = np.zeros(cam.fr.shape, dtype=np.uint8)
        # Only keep the key features of the card
        cv.fillConvexPoly(cardMsk, hull, (255, 255, 255))
        cardMsk = np.where(card == np.array([255, 255, 255]), 0, cardMsk)
        # Paste the card image on the frame
        cam.fr = np.where(cardMsk == np.array([255, 255, 255]), card, cam.fr)
    return cam

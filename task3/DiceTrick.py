import cv2 as cv
import numpy as np
import sys

sys.path.append("../")

from Macros import fprint


def apply_dice_trick(cam):
    cam.stPrint("Start Dice Trick")
    # Change to HSV Color Space
    hsv = cv.cvtColor(cam.fr, cv.COLOR_BGR2HSV)
    # Get white points of the frame
    light = max(hsv[:, :, 2].flatten())
    light = 200 if light < 200 else light
    satur = 30
    # Create a mask for the white points
    lowerb = np.array([0, 0, light - 10])
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
        dice = cv.imread("img/dices/" + str(cam.dice) + ".png")  # Read the dice image
        dice = cv.resize(dice, (cam.fr.shape[1], cam.fr.shape[0]))
        cardH, cardW, _ = dice.shape
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
        dice = cv.warpPerspective(dice, Mtx, (cam.fr.shape[1], cam.fr.shape[0]))
        # Make a mask for the card
        diceMsk = np.zeros(cam.fr.shape, dtype=np.uint8)
        # Let the mask set 255 where the square that founded by the corners
        cv.fillConvexPoly(diceMsk, pts2.astype(np.int32), (255, 255, 255))
        # Condense area by padding to black
        diceMsk = cv.bitwise_not(diceMsk)  # Reverse the mask
        diceMsk = cv.dilate(diceMsk, np.ones((4, 4), np.uint8), iterations=1)
        diceMsk = cv.bitwise_not(diceMsk)  # Reverse the mask back
        diceMsk = np.where(dice == np.array([255, 255, 255]), 20, diceMsk)
        # Paste the card image on the frame
        cam.fr = np.where(diceMsk == np.array([255, 255, 255]), dice, cam.fr)
        cam.fr = np.where(diceMsk == np.array([20, 20, 20]), light, cam.fr)
    return cam

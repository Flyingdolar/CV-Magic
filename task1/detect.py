import cv2 as cv
import numpy as np
import mediapipe as mp
from typing import Tuple, Union


# Define Coin_Detection Function
def coinDet(
    frame: np.ndarray, draw: bool = False
) -> Tuple[Union[np.ndarray, None], np.ndarray]:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9, 9), 0)
    circles = cv.HoughCircles(
        image=gray,
        method=cv.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=50,
        minRadius=10,
        maxRadius=100,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles[0, :, 2] += 5
        # Draw Circles
        if draw:
            for circle in circles[0, :]:
                cv.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)
        return circles[0, :], frame
    else:
        return None, frame


# Define Hand_Detection Function
def handDet(
    frame: np.ndarray, draw: bool = False
) -> Tuple[Union[np.ndarray, None], np.ndarray]:
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    results = hands.process(frame)
    # IF No Hand is Detected
    if results.multi_hand_landmarks is None:
        return None, frame
    lMarks = results.multi_hand_landmarks[0].landmark

    # Get the Key Points of the Hand on the Frame
    imgRow, imgCol, _ = frame.shape
    keyPoints = []
    for lm in lMarks:
        keyPoints.append((int(lm.x * imgCol), int(lm.y * imgRow)))

    # Get the Polygon of the Hand
    contours = cv.convexHull(np.array(keyPoints))
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv.drawContours(mask, [contours], -1, (255, 255, 255), -1)

    # Get the Contours of the Hand
    if (lMarks is not None) and draw:
        # Draw Polygon of the Hand, filled with green
        cv.drawContours(frame, [contours], -1, (0, 255, 0), 1)

    # Return the Contours and the Frame
    return mask, frame


# Define Whether Hand is Touching Coin
def touchCk(handMsk: np.ndarray, coins: np.ndarray) -> list:
    touchList = []  # List of Coins that are Touching the Hand
    if coins is None or handMsk is None:
        return touchList  # No Coin or Hand Detected

    for coin in coins:  # Detect if Hand is Touching which Coin
        if handMsk[coin["y"], coin["x"]][0] == 255:
            touchList.append(coin)
    return touchList


# Define Hide Coin Function (Notice: Do not cover the hand part.)
def coinHd(frame: np.ndarray, handMsk: np.ndarray, coins: np.ndarray) -> np.ndarray:
    coinPoints = []
    in_contour = False
    bg = cv.imread("tmp/background.png")

    # Create a mask image by filling the coin area with white
    mask = np.zeros(frame.shape, dtype=np.uint8)
    for coin in coins:
        if coin["ts"] % 2 == 1:
            cv.circle(mask, (coin["x"], coin["y"]), coin["r"], (255, 255, 255), -1)

    # Restore the hands in the mask image(if exists)
    if handMsk is not None:
        mask = np.where(handMsk == np.array([255, 255, 255]), 0, mask)
    # Filled the frame with the background image where the mask is white
    frame = np.where(mask == np.array([255, 255, 255]), bg, frame)
    return frame  # Return the frame

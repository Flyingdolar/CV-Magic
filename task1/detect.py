import cv2 as cv
import numpy as np
import mediapipe as mp
from typing import Tuple, Union


# Define Coin_Detection Function
def coin_detection(
    frame: np.ndarray, draw: bool = False
) -> Tuple[Union[np.ndarray, None], np.ndarray]:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    circles = cv.HoughCircles(
        image=gray,
        method=cv.HOUGH_GRADIENT,
        dp=1,
        minDist=5,
        param1=80,
        param2=80,
        minRadius=0,
        maxRadius=0,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Draw Circles
        if draw:
            for circle in circles[0, :]:
                cv.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)
        return circles, frame
    else:
        return None, frame


# Define Hand_Detection Function
def hand_detection(
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

    # Convert to HSV and set range
    hsvImg = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsvImg)
    minHSV = np.array([np.amin(h), np.amin(s), np.amin(v)])
    maxHSV = np.array([np.amax(h), np.amax(s), np.amax(v)])

    # Create Mask & Draw Contour
    mask = cv.inRange(hsvImg, minHSV, maxHSV)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Remove all Contours that are not in the Hand(Key Points)
    for contour in contours:
        for point in contour:
            if tuple(point[0]) not in keyPoints:
                cv.drawContours(mask, [contour], -1, (0, 0, 0), -1)

    if (lMarks is not None) and draw:
        cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # Return the Contours and the Frame
    return contours, frame


# Define Whether Hand is Touching Coin
def check_touch(handCts: np.ndarray, coins: np.ndarray) -> list:
    touchList = []  # List of Coins that are Touching the Hand
    if coins is None or handCts is None:
        return touchList  # No Coin or Hand Detected

    for coin in coins:  # Detect if Hand is Touching which Coin
        for handCt in handCts:
            # Test if Coins center point is in the Hand Contour
            if cv.pointPolygonTest(handCt, (coin.x, coin.y), False) >= 0:
                touchList.append(coin)  # Add Coin Index to List
    return touchList


# Define Hide Coin Function (Notice: Do not cover the hand part.)
def hide_coin(frame: np.ndarray, handCts: np.ndarray, coin: dict) -> np.ndarray:
    coinPoints = []
    in_contour = False
    bg = cv.imread("background.png")

    # Collect all the points in the coin area
    for i in range(coin.x - coin.r, coin.x + coin.r):
        for j in range(coin.y - coin.r, coin.y + coin.r):
            if (i - coin.x) ** 2 + (j - coin.y) ** 2 <= coin.r**2:
                coinPoints.append((i, j))

    # Fill the coin area with the image background
    for point in coinPoints:
        if handCts is not None:
            for handCt in handCts:
                if cv.pointPolygonTest(handCt, point, False) >= 0:
                    in_contour = True
                    break
        if not in_contour:
            frame[point[1], point[0]] = bg[point[1], point[0]]
    return frame

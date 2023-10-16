import cv2 as cv
import numpy as np
import mediapipe as mp


# Define Coin_Detection Function
def coin_detection(frame, draw=False):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    circles = cv.HoughCircles(
        image=gray,
        method=cv.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Draw Circles
        if draw:
            for idx in circles[0, :]:
                # Draw Outer Circle
                cv.circle(frame, (idx[0], idx[1]), idx[2], (0, 255, 0), 2)
                # Draw Center of Circle
                cv.circle(frame, (idx[0], idx[1]), 2, (0, 0, 255), 3)
        return circles[0, :], frame
    else:
        return None, frame


# Define Hand_Detection Function
def hand_detection(frame, draw=False):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    results = hands.process(frame)

    if draw:
        mpDraw = mp.solutions.drawing_utils

        # Convert the BGR image to RGB before processing
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Draw the hand annotations on the image
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLandmarks, mpHands.HAND_CONNECTIONS)

        # Convert the RGB image back to BGR for display
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # Return the Position of the Hand, and the Frame
    return results, frame


# Define Whether Hand is Touching Coin
def isTouching(hands, coins):
    if coins is None or hands is None:
        return False
    if hands.multi_hand_landmarks:
        for handLandmarks in hands.multi_hand_landmarks:
            for coin in coins:
                for landmark in handLandmarks.landmark:
                    if (
                        landmark.x * 640 > coin[0] - coin[2]
                        and landmark.x * 640 < coin[0] + coin[2]
                        and landmark.y * 480 > coin[1] - coin[2]
                        and landmark.y * 480 < coin[1] + coin[2]
                    ):
                        return True
    return False

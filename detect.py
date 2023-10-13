import cv2 as cv
import numpy as np
import mediapipe as mp


# Define Coin_Detection Function
def coin_detection(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        1,
        100,
        param1=80,
        param2=30,
        minRadius=0,
        maxRadius=100,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0]
    else:
        return None


# Define Hand_Detection Function
def hand_detection(frame, draw=False):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if draw:
        mpDraw = mp.solutions.drawing_utils

        # Convert the BGR image to RGB before processing
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame)

        # Draw the hand annotations on the image
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLandmarks, mpHands.HAND_CONNECTIONS)

        # Convert the RGB image back to BGR for display
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # Return the Position of the Hand, and the Frame
    return hands, frame

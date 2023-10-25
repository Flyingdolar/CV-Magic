import cv2 as cv
import sys

sys.path.append("../")  # Add parent directory to Python Path

from Macros import fprint  # Import Print Function
import task1.Detect as det  # Import Detection Function


def apply_coin_trick(cap):
    if cap.stage == 0:  # Capture Background
        cap.stPrint("Capture Background")
        if cv.waitKey(1) == ord(" "):
            cap.bg = cap.fr.copy()
            cv.imwrite("tmp/background.png", cap.bg)
            cap.chStage()
    elif cap.stage == 1:  # Capture Coin Position
        cap.stPrint("Capture Coin Position")
        cpList, cap.fr = det.coinDet(cap.fr, draw=True)  # Detect Coin
        if cv.waitKey(1) == ord(" "):
            if cpList is not None:
                cap.cpList = [
                    {"ts": 0, "hv": False, "x": cPos[0], "y": cPos[1], "r": cPos[2]}
                    for cPos in cpList
                ]
                cv.imwrite("tmp/coin.png", cap.fr)
                cap.chStage()  # Go to next stage
            else:
                fprint("E", "No Coin Detected!")
                cap.chStage(1)  # Restart this stage
    elif cap.stage == 2:  # Start Coin Trick
        cap.stPrint("Start Coin Trick")
        hdMsk, cap.fr = det.handDet(cap.fr, draw=True)
        tCoins = det.touchCk(hdMsk, cap.cpList)
        if tCoins is not None:  # If there is a coin touching the hand
            for coin in cap.cpList:  # For each coin change the state
                if coin in tCoins and coin["hv"] is False:
                    coin["ts"] += 1  # Add 1 to the times of touch
                    coin["hv"] = True  # Set hover to True
                elif coin not in tCoins and coin["hv"] is True:
                    coin["hv"] = False  # Set hover to False
        cap.fr = det.coinHd(cap.fr, hdMsk, cap.cpList)
    else:
        cap.stPrint("Unknown Stage")
    return cap

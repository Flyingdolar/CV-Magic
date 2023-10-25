import cv2 as cv  # Import OpenCV for Image Manipulation
import numpy as np
import argparse  # Import Argument Parser for Command Line Arguments

from Macros import fprint  # Import Print Function
from task1.CoinTrick import apply_coin_trick  # Import Process Function
from task2.CardTrick import apply_card_trick  # Import Process Function
from task3.DiceTrick import apply_dice_trick  # Import Process Function

trick_list = [  # List of Trick Functions this program can apply
    apply_coin_trick,  # 1: Coin Trick
    apply_card_trick,  # 2: Card Trick
    apply_dice_trick,  # 3: Dice Trick
]


def parse_args() -> argparse.Namespace:  # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Magic Trick Argument Parser")
    parser.add_argument(
        "--file_path",
        type=str,
        default="",
        help="Using File Mode --> Path to Video File",
        required=False,
    )
    parser.add_argument(
        "--cam_num",
        type=int,
        default=0,
        help="Using Camera Mode --> Camera Number",
        required=False,
    )
    parser.add_argument(
        "--trick",
        type=str,
        default="apply_coin_trick",
        help="Select the Trick Name that want to be Implemented",
        required=False,
    )
    return parser.parse_args()


class useCamera:  # Class for Camera Operations
    def __init__(
        self,
        method: str,
        file_path="",
        cam_num=0,
    ) -> None:
        self.method = method  # Trick to Apply
        if self.method is None:  # Trick not Found
            fprint("E", "Trick not Found!")
            exit()
        self.filePath = file_path  # Path to Video File
        self.camNum = cam_num  # Camera Number
        if file_path == "":  # Use Camera
            fprint("M", "Opening Camera...")
            self.cam = cv.VideoCapture(cam_num)
        else:  # Use Video File
            fprint("M", "Opening Video File...")
            print(file_path)
            self.cam = cv.VideoCapture(file_path)
        if not self.cam.isOpened():
            fprint("E", "Cannot open camera or video file")
            exit()
        fprint("M", "Camera Opened!")

        # Close Camera Exposure
        self.cam.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cam.set(cv.CAP_PROP_EXPOSURE, 50.0)

        # Lower Camera Resolution
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        # Lower Camera FPS
        self.cam.set(cv.CAP_PROP_FPS, 12)

        # Variables for Camera
        self.frNum = 0  # Count for frame
        self.stage = 0  # Stage of the program
        self.stPrinted = False  # Check if stage is printed
        # Other Variables
        self.bg = None  # Background Image
        self.cpList = None  # Coins Position List

    def read(self) -> None:  # Read Camera
        ret, frame = self.cam.read()
        if not ret:
            fprint("E", "Can't receive frame (stream end?). Exiting ...")
            self.cam.release(), cv.destroyAllWindows()
            exit()
        self.ret, self.fr = ret, frame

    def chStage(self, stage=-1) -> None:  # Change Stage
        self.stage = stage if stage != -1 else self.stage + 1
        self.stPrinted = False

    def stPrint(self, message: str) -> None:  # Print Message for Stage
        if not self.stPrinted:
            fprint("C", "<-- " + message + " -->")
            self.stPrinted = True

    def process(self) -> None:  # Process Frame
        if self.method == "apply_coin_trick":
            self = apply_coin_trick(self)
        elif self.method == "apply_card_trick":
            self = apply_card_trick(self)
        elif self.method == "apply_dice_trick":
            self = apply_dice_trick(self)

    def __del__(self):  # Release Camera & Destroy Windows
        self.cam.release(), cv.destroyAllWindows()


def main():
    args = parse_args()  # Parse Command Line Arguments
    if args.trick is None:  # Choose Trick to Apply
        fprint("M", "Choose Trick to Apply:")
        fprint("M", "1: Coin Trick | 2: Card Trick | 3: Dice Trick")
        fprint("M", "0: Exit")
        while True:
            keyNum = int(input("Your Choice: "))
            if 1 <= keyNum <= 3:  # Setup Trick
                args.trick = trick_list[keyNum - 1]
                break
            elif keyNum == 0:  # Handle Exit
                exit()

    # Open & Setup Camera
    cap = useCamera(args.trick, args.file_path, args.cam_num)
    # Start Reading Video
    while True:
        cap.read()  # Read Camera
        cap.process()  # Process Frame
        cv.imshow("frame", cap.fr)  # Display frame
        usrKey = cv.waitKey(1)
        if usrKey == ord("w"):  # Press 'w' to Increase Exposure
            cap.cam.set(cv.CAP_PROP_EXPOSURE, cap.cam.get(cv.CAP_PROP_EXPOSURE) - 1)
            print("Exp = ", cap.cam.get(cv.CAP_PROP_EXPOSURE))
        elif usrKey == ord("s"):  # Press 's' to Adjust Exposure Down
            cap.cam.set(cv.CAP_PROP_EXPOSURE, cap.cam.get(cv.CAP_PROP_EXPOSURE) + 1)
            print("Exp = ", cap.cam.get(cv.CAP_PROP_EXPOSURE))
        elif usrKey == ord("q"):  # Press 'q' to Quit
            print("Quit")
            break

    # Release Camera & Destroy Windows
    del cap


if __name__ == "__main__":
    main()  # Run Main Function

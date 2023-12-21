import numpy as np
import cv2 as cv

imgArray = np.array(
    [
        [0, 0, 0, 0, 255, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 255, 0],
        [0, 255, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 0, 255, 0, 0],
    ]
)

# Let y=ax+b, for range a=[-2,2], b=[0,6]
a = np.linspace(-2, 2, 5)
b = np.linspace(0, 6, 7)

hough = np.zeros((len(b), len(a)))

for adx, a_val in enumerate(a):
    for bdx, b_val in enumerate(b):
        for x, y in zip(*np.where(imgArray == 255)):
            if y == a_val * x + b_val:
                hough[bdx, adx] += 1

print(hough)

import cv2 as cv
import numpy as np

# Image Array
img = np.array(
    [
        [240, 232, 81, 55, 154, 202, 139, 188, 199],
        [201, 188, 44, 28, 220, 238, 175, 225, 74],
        [168, 13, 233, 24, 0, 55, 77, 43, 203],
        [100, 55, 132, 101, 144, 176, 175, 158, 76],
        [195, 175, 23, 79, 15, 55, 145, 13, 54],
        [101, 183, 140, 149, 108, 150, 215, 32, 98],
        [103, 202, 194, 130, 151, 250, 124, 161, 218],
        [246, 186, 180, 141, 32, 48, 211, 144, 232],
        [213, 48, 217, 160, 2, 16, 189, 67, 132],
    ]
)

# The Center Pixel Value
center = img[4, 4]
print("Center Pixel Value:", center)

# Calculate the Gradient Direction and Magnitude of the Center Pixel
grad_x = np.gradient(img, axis=1)
grad_y = np.gradient(img, axis=0)
magnitude = np.sqrt(grad_x**2 + grad_y**2)
direction = np.arctan2(grad_y, grad_x)
center_pixel_direction = direction[4, 4]
center_pixel_magnitude = magnitude[4, 4]
# Turn the Direction into 0 to 180 Degrees
center_pixel_direction = np.abs(center_pixel_direction) * 180 / np.pi
print("Center Pixel Direction:", center_pixel_direction)
print("Center Pixel Magnitude:", center_pixel_magnitude)

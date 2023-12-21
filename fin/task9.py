import numpy as np


def _gradient(image):
    # Use Sobel operator to get gradient
    grad_x = np.gradient(image, axis=1)
    grad_y = np.gradient(image, axis=0)
    # Get magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    return magnitude, direction


def _hogDesript(ditTan):
    bins = np.arange(0, 181, 20)  # Set bins
    ditTan = np.abs(ditTan) * 180 / np.pi  # Turn radian to degree
    # Calculate weights
    weights, _ = np.histogram(ditTan, bins=bins)
    return weights


def getBlockHog(image):
    mag, dir = _gradient(image)  # Get magnitude and direction
    blockDirection = []
    # Get direction of each block
    for i in range(1, 8, 3):
        for j in range(1, 8, 3):
            blockDirection.append(dir[i, j])
    # Calculate HOG feature vector
    hog_descriptor = _hogDesript(blockDirection)
    return hog_descriptor


# 定義圖像
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

# Get HOG feature vector
hogFeat = getBlockHog(img)

# Print the HOG feature vector
print("HOG Feature Vector (without normalization):")
print(hogFeat)

# Normalize HOG feature vector
hogFeat = hogFeat / 9

# Print the normalized HOG feature vector
print("HOG Feature Vector (with normalization):")
print(hogFeat)

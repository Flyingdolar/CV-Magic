import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2


def calculate_hog_descriptor(image):
    # 計算 HOG 特徵描述符
    # `orientations` 定義了特徵描述符的方向數量，`pixels_per_cell` 定義了每個單元格的像素數量
    # `cells_per_block` 定義了每個塊的單元格數量
    features, hog_image = hog(
        image,
        orientations=8,
        pixels_per_cell=(3, 3),
        cells_per_block=(1, 1),
        block_norm="L2-Hys",
        visualize=True,
    )

    # 對 HOG 圖像進行對比度增強
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return features, hog_image_rescaled


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

# 計算 HOG 特徵描述符
hog_features, hog_image = calculate_hog_descriptor(img)

# 顯示結果
print("HOG Feature Descriptor:")
print(hog_features)

# 顯示原始圖像和 HOG 圖像
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("原始圖像")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap="gray")
plt.title("HOG 圖像")
plt.axis("off")

plt.show()

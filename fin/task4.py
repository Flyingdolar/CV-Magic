import cv2 as cv
import numpy as np

# Find the secret text by edge detection
ori_img = cv.imread("taiwan2.png")

# Find the edge in 2 ways
ori_img = cv.cvtColor(ori_img, cv.COLOR_BGR2GRAY)
edge1 = cv.Canny(ori_img, 0, 0, apertureSize=3)
edge2 = cv.Canny(ori_img, 0, 100, apertureSize=3)

# Let edge1 - edge2
edge = edge1 - edge2

# Show the edge
cv.imshow("edge1", edge1)
cv.waitKey(0)
cv.imshow("edge", edge)
cv.waitKey(0)
cv.destroyAllWindows()

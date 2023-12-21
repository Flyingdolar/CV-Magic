import cv2 as cv
import numpy as np

# 1. Read the image into your program,
img = cv.imread("abstract_lines.png")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 2. apply the Canny edge detector with parameters min=150 and max=500,
edges = cv.Canny(img, 150, 500, apertureSize=3)
# 3. apply the HoughLines algorithm with a resolution of 5.0 deg for theta. and a resolution of 3.0 for rho. Set the threshold parameter to 100.
lines = cv.HoughLines(edges, 3.0, np.pi / 180 / 5.0, 100)

# Only preserve the line that degree is between 30 and 60
lines = [line for line in lines if 30 < line[0][1] / np.pi * 180 < 60]

# Print the number of lines detected
print(len(lines))

# Draw the lines on the image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho * a
    y0 = rho * b
    x1 = int(x0 + 1000 * -b)
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * -b)
    y2 = int(y0 - 1000 * a)
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Show the image
cv.imshow("lines", img)
cv.waitKey(0)
cv.destroyAllWindows()

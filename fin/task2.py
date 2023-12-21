import cv2 as cv
import numpy as np

# Somebody put text by implementing the following code:
# text_img = cv.putText( text_img, secret, (x,y), 1, 1, (255,255,0), 5 )

# And also he hide the text by implementing the following code:
# new_img = alpha * text_img + (1 - alpha) * img

# Now I get the new image only, how can I get the secret text?
new_img = cv.imread("taiwan2.png")

# Use the Edge Detection to get the secret text
gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 0, 0, apertureSize=3)
lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

# Show the edges and lines
cv.imshow("edges", edges)
cv.waitKey(0)
cv.destroyAllWindows()

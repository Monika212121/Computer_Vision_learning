# Draw lines

import cv2 as cv
import numpy as np

RED = (255, 0, 255)
BLUE = (255, 255, 0)

p0, p1, p2 = (10, 10), (100, 150), (400, 20)

# Task1:  Create a color image
image = np.zeros((200, 500, 3), np.uint8)

cv.line(image, p0, p1, RED, 2)
cv.line(image, p1, p2, BLUE, 10)
cv.imshow('BGR IMAGE', image)

# ---------------------------------------------------------------------------

# Task2: Create a black and white image
grey_image = np.zeros((200, 500), np.uint8)

p3 = (400, 150)
cv.line(grey_image, p0, p1, 80, 5)
cv.line(grey_image, p1, p3, 200, 3)
cv.imshow('GREYSCALE IMAGE', grey_image)

cv.waitKey(0)
cv.destroyAllWindows()
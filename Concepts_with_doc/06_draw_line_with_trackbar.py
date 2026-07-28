# Drawing line using trackbar

import cv2 as cv
import numpy as np

BLUE = (255, 0, 0)
p0, p1 = (100, 20), (400, 100)


def trackbar(x):
    x = max(1, x)
    cv.displayOverlay('window', f'thickness = {x}')
    image[:] = 0                                # again taking image as a fresh black screen
    cv.line(image, p0, p1, BLUE, x)
    cv.imshow('window', image)


image = np.zeros((150, 450, 3), np.uint8)
cv.line(image, p0, p1, BLUE, 2)
cv.imshow('window', image)
cv.createTrackbar('thickness', 'window', 5, 100, trackbar)


cv.waitKey(0)
cv.destroyAllWindows()
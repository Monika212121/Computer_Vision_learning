# Convert color image to grayscale and save.

import cv2 as cv

image = cv.imread('D:/Computer_Vision_learning/Concepts_with_doc/dataset/messi.jpg')
cv.imshow('Color image', image)                                     # display color image

greyImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Grey color image', greyImage)                            # display grey image

cv.imwrite('grey_image.png', greyImage)                # Save grey image

cv.waitKey()
cv.destroyAllWindows()
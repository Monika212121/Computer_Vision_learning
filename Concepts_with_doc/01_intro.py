import cv2 as cv

image = cv.imread('D:/Computer_Vision_learning/Concepts_with_doc/dataset/messi.jpg')

cv.imshow('See this image!!!!!!!', image)

cv.waitKey()
cv.destroyAllWindows()
import cv2 as cv

image = cv.imread('D:/Computer_Vision_learning/Concepts_with_doc/dataset/messi.jpg')

#image[350:450, 300:400] = (0, 0, 255)

image2 = image[200: 400, 200: 400]

cv.imshow('display cropped image', image2)

cv.waitKey(0)
cv.destroyAllWindows()

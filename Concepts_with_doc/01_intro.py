# Loading and displaying an image

import cv2 as cv

image = cv.imread('D:/Computer_Vision_learning/Concepts_with_doc/dataset/messi.jpg')
cv.imshow('See this image!!!!!!!', image)

# cv.waitKey()                                      # param = 0 / None means the function waits indefinitely 

cv.waitKey(5000)                                    # param = 5000 means the function waits for 5000 milli seconds before executing next line of code
cv.destroyAllWindows()                              # image will be displayed for 5000 milli seconds
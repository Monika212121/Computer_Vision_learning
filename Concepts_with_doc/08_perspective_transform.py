# Aim: Implementing Perspective Transform 

import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)


while True:
    ret, frame = cap.read()

    # Locate points on document or object which we want to transform
    #points1 = np.float32([[0,100], [300,100], [0,200], [300,100]])
    #points2 = np.float32([[0,0], [50,0], [0,30], [30,50]])

    # Locate points of the documents
    # or object which you want to transform
    points1 = np.float32([[0, 260], [640, 260], [0, 400], [640, 400]])
    points2 = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])

    # Apply Perspective Transform Algorithm
    matrix = cv.getPerspectiveTransform(points1, points2)
    result = cv.warpPerspective(frame, matrix, (800, 600))

    # Wrap the transformed image
    cv.imshow('Initial capture', frame)
    cv.imshow('Transformed capture', result)

    if cv.waitKey(20) == 27:
        break


cap.release()
cv.destroyAllWindows()

# Reading and displaying a video stream

import cv2 as cv

capture = cv.VideoCapture(0)        

while True:
    ret, frame = capture.read()

    cv.imshow('Video stream', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv.destroyAllWindows()
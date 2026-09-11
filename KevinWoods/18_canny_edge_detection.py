import os
import cv2 as cv



def callback():
    pass



def cannyEdge():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    winName = "Canny Edge Detection"
    cv.namedWindow(winname= winName)
    cv.createTrackbar('minThreshold', winName, 0, 255, callback)
    cv.createTrackbar('maxThreshold', winName, 0, 255, callback)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        minThresh= cv.getTrackbarPos('minThreshold', winName)
        maxThresh= cv.getTrackbarPos('maxThreshold', winName)

        # Implement Canny Edge Detection
        cannyEdge = cv.Canny(img, minThresh, maxThresh) 
        cv.imshow(winName, cannyEdge)

        output_path = os.path.join(root, 'output/18_cannyEdgeDetection.jpg') 
        cv.imwrite(output_path, cannyEdge)


    cv.destroyAllWindows()



if __name__ == "__main__":
    cannyEdge()



# Note:

# Here I took minThresh = 102 and maxThresh = 204, for the car image. It giving maximum accurate edges.
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def callback(input):
    pass



def averageFiltering():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    winName = "Average Filter"
    cv.namedWindow(winname= winName)
    cv.createTrackbar('n', winName, 1, 100, callback)

    h, w, _ = img.shape
    scale = 2

    new_w, new_h = int(w * scale), int(h * scale)

    new_img = cv.resize(img, (new_w, new_h))


    while True:

        if cv.waitKey(1) == ord('q'):
            break

        # Getting kernal size from trackbar
        n = cv.getTrackbarPos('n', winName)

        # Average Filtering according to the dynamic Kernal size
        imgFilter = cv.blur(new_img, (n,n))

        cv.imshow(winName, imgFilter)

        output_path = os.path.join(root, 'output/08_AverageFiltering.jpg') 
        cv.imwrite(output_path, imgFilter)


    cv.destroyAllWindows()



if __name__ == "__main__":
    averageFiltering()
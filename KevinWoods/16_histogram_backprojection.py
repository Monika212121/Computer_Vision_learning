# INCOMPLETE

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def histBackPropagation():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    plt.figure(figsize = (12,6))

    plt.subplot(231)
    plt.imshow(img)
    plt.title('Original image')

    # 2. Take a  specific region in the image
    imgRegion = img[270:305, 380:440, :]
    plt.subplot(232)
    plt.imshow(imgRegion)
    plt.title('Region in car bonnet')

    # 3. HSV and Hist of the specific region in the image
    imgRegionHSV = cv.cvtColor(imgRegion, cv.COLOR_RGB2HSV)
    imgRegionHist = cv.calcHist([imgRegionHSV], [0,1], None, [180,256], [0,180,0,256])
    cv.normalize(imgRegionHist, imgRegionHist, 0, 255, cv.NORM_MINMAX)

    # Implemneting Histogram backpropagation
    imgHSV = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    output = cv.calcBackProject([imgHSV], [0,1], imgRegionHist, [0,180,0,256], 1)

    plt.subplot(233)
    plt.imshow(output)

    
    plt.show()


if __name__ == "__main__":
    histBackPropagation()



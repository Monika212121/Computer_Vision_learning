 import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def histogram2D():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Drawing 2D Histogram (H on y-axis and S on x-axis)
    hist = cv.calcHist([imgHSV], [0,1], None, [180,256], [0,180,0,256])

    plt.figure()
    plt.subplot(141)
    plt.imshow(imgRGB)
    plt.title('RGB image')

    plt.subplot(142)
    plt.imshow(hist)
    plt.ylabel('hue')
    plt.xlabel('saturation')
    plt.title('Histogram')

    # NOTE: These H, S, V values are taken from the histogram bright section
    lowerBound = np.array([90, 0, 0])                       # H, S, V
    upperBound = np.array([110, 40, 255])                   # H, S, V
    mask = cv.inRange(imgHSV, lowerBound, upperBound)

    plt.subplot(143)
    plt.imshow(imgHSV)
    plt.title('HSV image')

    plt.subplot(144)
    plt.imshow(mask, cmap= 'gray')   
    plt.title('Binary mask')

    output_path = os.path.join(root, 'output/06_histogram2D.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()




if __name__ == '__main__':
    histogram2D()


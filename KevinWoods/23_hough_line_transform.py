# NOTE: Our main aim here is to identify the main line and extract it out.

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def houghLineTransform():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car2.webp')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    imgBlur  = cv.GaussianBlur(img, (21,21), 3)
    cannyEdge = cv.Canny(imgBlur, 50, 100)

    plt.figure()
    plt.subplot(141)
    plt.imshow(img)
    plt.title('Original')

    plt.subplot(142)
    plt.imshow(imgBlur)
    plt.title('Gaussian Blurred')

    plt.subplot(143) 
    plt.imshow(cannyEdge)
    plt.title('Canny Edge detected')

    distRes = 1
    angleRes = np.pi / 180
    threshold = 150
    lines = cv.HoughLines(cannyEdge, distRes, angleRes, threshold)
    print("Lines: ", lines)

    k = 3000

    for curLine in lines:
        rho, theta = curLine[0]
        dhat = np.array([[np.cos(theta)], [np.sin(theta)]])
        d = rho * dhat

        lhat = np.array([[-np.sin(theta)], [np.cos(theta)]])

        p1 = d + k*lhat
        p2 = d - k*lhat

        p1 = p1.astype(int)
        p2 = p2.astype(int)

        # points p1 and p2 coordinates are taken like this because they are a column matrix of size(2*1), x = mat[0][0], y = mat[1][0]
        cv.line(img, (p1[0][0], p1[1][0]), (p2[0][0], p2[1][0]), (255, 0, 0), 10)


    plt.subplot(144)
    plt.imshow(img)
    plt.title('Lines detected with threshold = 150')

    output_path = os.path.join(root, 'output/23_2_threshold-150.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')


    plt.show()



if __name__ == "__main__":
    houghLineTransform()

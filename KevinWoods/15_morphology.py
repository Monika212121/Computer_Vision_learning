import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def morphTransformation():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    imgGRAY = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    maxVal = 255
    blockSize = 7
    offsetC = 3

    plt.figure(figsize = (12, 6))

    plt.subplot(241)
    imgGaus = cv.adaptiveThreshold(imgGRAY, maxVal, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, offsetC)

    # 1. Thresholded image is granier, so applying Blurring to smooth the image.
    imgGausBlurred = cv.GaussianBlur(imgGaus, (7,7), sigmaX= 2)
    plt.imshow(imgGausBlurred, cmap= 'gray')
    plt.title('Gaussian Adaptive threshold')

    kernel = np.ones((7,7), np.uint8)

    # 2. Erosion (RESULT: Black is more emphasized)
    erosion = cv.erode(imgGausBlurred, kernel, iterations= 1)
    plt.subplot(242)
    plt.imshow(erosion, cmap= 'gray')
    plt.title('Erosion')

    # 3. Dilation (RESULT: White is more emphasized)
    dilation = cv.dilate(imgGausBlurred, kernel, iterations= 1)
    plt.subplot(243)
    plt.imshow(dilation, cmap= 'gray')
    plt.title('Dilation')

    # 4,5,6,7,8 Different Morphology types
    morphTypes = [cv.MORPH_OPEN, cv.MORPH_CLOSE, cv.MORPH_GRADIENT, cv.MORPH_TOPHAT, cv.MORPH_BLACKHAT]
    morphTiles = ['Open', 'Close', 'Gradient', 'Tophat', 'Blackhat']

    for i in range(len(morphTypes)):
        morphedImg = cv.morphologyEx(imgGausBlurred, morphTypes[i], kernel)
        plt.subplot(2, 4, i+4)
        plt.imshow(morphedImg, cmap= 'gray')
        plt.title(morphTiles[i])


    # Some custom kernels
    ellipseKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    crossKernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))

    print("Ellipse Kernel: \n", ellipseKernel)
    print("\nCross Kernel: \n", crossKernel)

    output_path = os.path.join(root, 'output/15_morphology.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()

    

if __name__ == "__main__":
    morphTransformation()

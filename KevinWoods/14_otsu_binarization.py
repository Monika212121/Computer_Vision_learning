import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def otsuThresholding():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    imgGRAY = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    plt.figure(figsize = (12,6))

    plt.subplot(221)
    plt.imshow(imgGRAY, cmap= 'gray')
    plt.title('Original Grayscale')

    hist = cv.calcHist([imgGRAY], [0], None, [256], [0,256])
    plt.subplot(222)
    plt.plot(hist)
    plt.xlabel('Intensity')
    plt.ylabel('No. of Pixels')
    plt.title(f'Histogram')

    # Binary segmentation with self taken global threshold (from Histogram)
    globalThresh = 160
    _, imgThresh = cv.threshold(imgGRAY, globalThresh, 255, cv.THRESH_BINARY)
    plt.subplot(223)
    plt.imshow(imgThresh, cmap='gray')
    plt.title(f'Global threshold: {globalThresh}')

    # Implementing Otsu Threshold
    arbThresh = 0                                                   # Does not affect threshold calculation, its more like a placeholder
    otsuThresh, imgOtsu = cv.threshold(imgGRAY, arbThresh, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    plt.subplot(224)
    plt.imshow(imgOtsu, cmap= 'gray')
    plt.title(f'Otsu threshold: {otsuThresh}')

    print(f"Otsu threshold automatically calculated as: {otsuThresh}")

    output_path = os.path.join(root, 'output/14_otsuThresholding.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')
    plt.show()
    


if __name__ == "__main__":
    otsuThresholding()


# NOTE:
# Here Otsu threshold worked better than self taken global threshold value, the binary segmented image is clearer. Foreground and background are spearated better.
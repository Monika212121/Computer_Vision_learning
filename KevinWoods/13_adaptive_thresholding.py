import os
import cv2 as cv
import matplotlib.pyplot as plt



def adaptiveThresholding():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    imgGRAY = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    plt.figure(figsize= (12,6))

    plt.subplot(141)
    plt.imshow(imgGRAY, cmap= 'gray')
    plt.title('Original Greyscale')

    plt.subplot(142)
    _, imgThresh = cv.threshold(imgGRAY, 170, 255, cv.THRESH_BINARY)
    plt.imshow(imgThresh, cmap= 'gray')
    plt.title('Global Thresholding')

    blockSize = 7
    offsetC = 2
    plt.subplot(143)
    imgMean = cv.adaptiveThreshold(imgGRAY, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, offsetC)
    plt.imshow(imgMean, cmap= 'gray')
    plt.title('Mean adaptive thresholding')

    plt.subplot(144)
    imgGaus = cv.adaptiveThreshold(imgGRAY, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, offsetC)
    plt.imshow(imgGaus, cmap= 'gray')
    plt.title('Guassian adaptive thresholding')

    output_path = os.path.join(root, 'output/13_adaptiveThresholding.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')
    plt.show()




if __name__ == "__main__":
    adaptiveThresholding()



# NOTE:
# Both Mean and Guassian adaptive thersholded looks almost similar.
# The only difference is that the "Mean adaptive" thresholded image is granier than "Guassian adaptive" thresholded image.
# Gaussian image has finer patterns.
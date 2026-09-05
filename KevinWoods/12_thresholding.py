import os
import cv2 as cv
import matplotlib.pyplot as plt



def thresholding():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    imgGRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    hist = cv.calcHist([imgGRAY], [0], None, [256], [0, 256])

    plt.figure(figsize = (12, 6))
    plt.plot(hist)

    plt.xlabel('bins')
    plt.ylabel('No. of pixels')
    plt.title('Histogram')
    
    hist_output_path = os.path.join(root, 'output/12_1_gray_histogram.jpg') 
    plt.savefig(hist_output_path, bbox_inches = 'tight')


    threshOpts = [cv.THRESH_BINARY, cv.THRESH_BINARY_INV, cv.THRESH_TOZERO, cv.THRESH_TOZERO_INV, cv.THRESH_TRUNC]
    threshNames = ['Binary', ' Binary Inverted', 'ToZero', 'ToZero Inverted', 'Truncated']

    plt.figure(figsize= (12, 6))

    plt.subplot(231)
    plt.imshow(imgGRAY, cmap= 'gray')
    plt.title('Original Grayscale')

    for i in range(len(threshOpts)):
        plt.subplot(2, 3, i+2)

        _, imgThres = cv.threshold(imgGRAY, 160, 255, threshOpts[i])

        plt.imshow(imgThres, cmap= 'gray')
        plt.title(threshNames[i])



    output_path = os.path.join(root, 'output/12_2_thresholding.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()



if __name__ == "__main__":
    thresholding()


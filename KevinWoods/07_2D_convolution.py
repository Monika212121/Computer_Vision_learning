import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def Convolution2D():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Creating kernel
    n = 25
    kernel = np.ones((n,n), np.float32) / (n*n)

    # Smoothening / blurring an image
    imgFilter = cv.filter2D(imgRGB, -1, kernel)

    plt.figure(figsize = (12, 6)) 
    plt.subplot(131)
    plt.imshow(imgRGB)

    plt.subplot(132)
    plt.imshow(imgFilter)
    plt.title('Blurring with kernel size = 25')


    n2 = 50
    kernel2 = np.ones((n2,n2), np.float32) / (n2*n2)
    imgFilter2 = cv.filter2D(imgRGB, -1, kernel2)

    plt.subplot(133)
    plt.imshow(imgFilter2)
    plt.title('Blurring with kernel size = 50')


    output_path = os.path.join(root, 'output/07_2D_Convolution.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()




if __name__ == "__main__":
    Convolution2D()
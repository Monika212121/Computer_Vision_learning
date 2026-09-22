import os 
import cv2 as cv
import matplotlib.pyplot as plt



def harrisCorner():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(131)
    plt.imshow(imgGRAY, cmap= 'gray')
    plt.title('Original Grayscale')

    blockSize = 5
    sobelSize = 3
    k = 0.04
    harris = cv.cornerHarris(imgGRAY, blockSize, sobelSize, k)
    plt.subplot(132)
    plt.imshow(harris)
    plt.title('Harris')

    imgRGB[harris > 0.05*harris.max()] = [255, 0, 0]
    plt.subplot(133)
    plt.imshow(imgRGB)
    plt.title('Corners marked on original image')


    output_path = os.path.join(root, 'output/27_harrisCorner.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()



if __name__ == "__main__":
    harrisCorner()
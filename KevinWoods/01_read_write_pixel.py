import os
import cv2 as cv
import matplotlib.pyplot as plt



def readAndWriteSinglePixel():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/cat.jpg')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()


    eyePixel = imgRGB[100,150]
    imgRGB[100,150] = (0, 255, 0)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()
    debug = 1


if __name__ == '__main__':
    readAndWriteSinglePixel()
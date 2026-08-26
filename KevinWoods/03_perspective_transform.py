import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def perspectiveTransform():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/cat1.jpg')
    img = cv.imread(imgPath)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    height, width, _ = img.shape

    p1 = np.array([[615,398],
                   [671, ]])


if __name__ == "__main__":
    perspectiveTransform()
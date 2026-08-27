import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def perspectiveTransform():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/cat2.webp ')
    img = cv.imread(imgPath)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    p1 = np.array([[282,395],
                   [437,374],
                   [324,510],
                   [510,473]], dtype = np.float32)

    p2 = np.array([[0,0],
                   [100,0],
                   [0,200],
                   [100,200]], dtype = np.float32)

    T = cv.getPerspectiveTransform(p1, p2)
    imgTrans = cv.warpPerspective(img, T, (100, 200))

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.plot(p1[:,0], p1[:,1], 'r.')

    plt.subplot(122)
    plt.imshow(imgTrans)
    plt.plot(p2[:,0], p2[:,1], 'r.')

    plt.show()


if __name__ == "__main__":
    perspectiveTransform()


# Implementing Morphological Operations

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


image = cv.imread('D:/Computer_Vision_learning/Concepts_with_doc/dataset/messi.jpg')

# Definig a kernel (structuring element)
kernel = np.ones((5,5), np.uint8)

# Erosion
erosion = cv.erode(image, kernel, iterations= 1)

# Dilation
dilation = cv.dilate(image, kernel, iterations= 1)

# Opening (Erosion followed by Dilation)
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

# Closing (Dilation followed by Erosion)
closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)


# Displaying the results
titles = ['Original image', 'Erosion', 'Dilation', 'Opening', 'Closing']
image_list = [image, erosion, dilation, opening, closing]

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(image_list[i], cmap= 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()


cv.waitKey()
cv.destroyAllWindows()

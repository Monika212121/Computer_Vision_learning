import os
import cv2 as cv
import matplotlib.pyplot as plt



def bilateralFiltering():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Implemneting Bilateral Filtering
    imgFilter = cv.bilateralFilter(imgRGB, 25, 100, 100)

    plt.figure(figsize = (14,8))

    plt.subplot(131)
    plt.imshow(imgRGB)
    plt.title('Original image')

    plt.subplot(132)
    plt.imshow(imgFilter)
    plt.title('Bilateral Fitering with sigmacolor/space= 100')

    imgFilter2 = cv.bilateralFilter(imgRGB, 25, 50, 50)

    plt.subplot(133)
    plt.imshow(imgFilter2)
    plt.title('Bilateral Blurring with sigmacolor/space= 50')

    output_path = os.path.join(root, 'output/11_bilateralFiltering.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')
    plt.show()



if __name__ == "__main__":
    bilateralFiltering()


# NOTE:
# Bilateral Filtering kept the edges intact and smooth the inside parts(fillers) of the image.
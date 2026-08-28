import os
import cv2 as cv
import matplotlib.pyplot as plt



def histogramEqual():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/cat3.png')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    # 1. Histogram of dark  distorted image
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    hist = cv.calcHist([img], [0], None, [256], [0,256])

    cdf = hist.cumsum()
    cdf_norm = cdf * float(hist.max()) / cdf.max()

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap = 'gray')
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdf_norm, color = 'b')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('No. of pixels')

    # 2. Histogram of equlaized image
    equImg = cv.equalizeHist(img)
    equHist = cv.calcHist([equImg], [0], None, [256], [0, 256])
    equcdf = equHist.cumsum()
    equ_cdf_norm = equcdf * float(equHist.max()) / equcdf.max()

    plt.subplot(232)
    plt.imshow(equImg, cmap = 'gray')
    plt.subplot(235)
    plt.plot(equHist)
    plt.plot(equ_cdf_norm, color = 'b')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('No. of pixels')


    # 3. CLAHE
    claheObj = cv.createCLAHE(clipLimit = 5, tileGridSize = (8,8))
    claheImg = claheObj.apply(img)
    claheHist = cv.calcHist([claheImg], [0], None, [256], [0,256])
    clahe_cdf = claheHist.cumsum()
    clahe_cdf_norm = clahe_cdf* float(claheHist.max()) / clahe_cdf.max()

    plt.subplot(233)
    plt.imshow(claheImg, cmap = 'gray')
    plt.subplot(236)
    plt.plot(claheHist)
    plt.plot(clahe_cdf_norm, color = 'b')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('No. of pixels')


    output_path = os.path.join(root, 'output/05_histEqualization.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')
    plt.show()



if __name__ == "__main__":
    histogramEqual()
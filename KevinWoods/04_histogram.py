import os
import cv2 as cv
import matplotlib.pyplot as plt



def greyHistogram():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/cat2.webp')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    plt.figure()
    plt.imshow(img, cmap='gray')

    hist = cv.calcHist([img], [0], None, [256], [0,256])

    plt.figure()
    plt.plot(hist)
    plt.xlabel('bins')
    plt.ylabel('No. of pixels')

    output_path = os.path.join(root, 'output/04_1_greyHist.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()



def colorHistogram():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/cat2.webp')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)

    colors = ['b', 'g', 'r']

    plt.figure()

    for i in range(len(colors)):
        hist = cv.calcHist([imgRGB], [i], None, [256], [0,256])
        plt.plot(hist, colors[i])


    plt.xlabel('Pixel intensity')
    plt.ylabel('No. of pixels')

    output_path = os.path.join(root, 'output/04_2_colorHist.jpg') 

    # Save the plot as an image file
    plt.savefig(output_path, bbox_inches = 'tight')
    plt.close()

    plt.show()      


# Same function as colorHistogram(), it is just for a specific region in the image.
def colorHistogramForRegion():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/cat2.webp')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # required region
    imgRGB = imgRGB[433:472, 324:367, :]
    region_output_path = os.path.join(root, 'output/04_3A_Region.jpg') 

    # Save the region plot
    cv.imwrite(region_output_path, imgRGB)

    plt.figure()
    plt.imshow(imgRGB)

    colors = ['b', 'g', 'r']

    plt.figure()

    for i in range(len(colors)):
        hist = cv.calcHist([imgRGB], [i], None, [256], [0,256])
        plt.plot(hist, colors[i])


    plt.xlabel('Pixel intensity')
    plt.ylabel('No. of pixels')

    output_path = os.path.join(root, 'output/04_3B_colorHistForRegion.jpg') 

    # Save the region's histogram plot
    plt.savefig(output_path, bbox_inches = 'tight')
    plt.close()

    plt.show()      




if __name__ == "__main__":
    #greyHistogram()
    #colorHistogram()
    colorHistogramForRegion()


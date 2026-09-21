import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def watershedSegmentation():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    img = img[122:180, 25:85]
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    plt.figure(figsize = (15, 15))

    # 1. Original grayscale
    plt.subplot(331)
    plt.imshow(imgGRAY, cmap = 'gray')
    plt.title ('Original Grayscale')

    # 2. Thresholding
    _, imgThresh = cv.threshold(imgGRAY, 50, 255, cv.THRESH_BINARY_INV)
    plt.subplot(332)
    plt.imshow(imgThresh, cmap = 'gray')
    plt.title('Binary mask')

    # 3. Dilation
    # NOTE: Dilation removes the black area in middle of an object and make categorization of objects better, reducing sharp details of edges. 
    kernel = np.ones((3,3), np.uint8)
    imgDilate = cv.morphologyEx(imgThresh, cv.MORPH_DILATE, kernel)            
    plt.subplot(333)
    plt.imshow(imgDilate)
    plt.title('Dilated image')

    # 4. Distance Transform
    disTrans = cv.distanceTransform(imgDilate, cv.DIST_L2, 5)
    plt.subplot(334)
    plt.imshow(disTrans)
    plt.title('Distance Transform')

    # 5. Threshold Distance transform
    # Drawing Histogram to find threshold value for "disThresh" image
    '''
    plt.figure()
    hist = cv.calcHist([disTrans], [0], None, [256], [0,256])
    plt.plot(hist, 'r')
    plt.show()
    '''
    _, distThresh = cv.threshold(disTrans, 4.75, 255, cv.THRESH_BINARY)
    plt.subplot(335)
    plt.imshow(distThresh)
    plt.title('Binary mask of Distance Transform')
    
    # 6. Connecting components 
    # NOTE: It enables us to find different regions and number them into different areas, using labels
    distThresh = np.uint8(distThresh)
    num, labels = cv.connectedComponents(distThresh)
    plt.subplot(336)
    plt.imshow(labels)
    plt.title('Connected Components Labels')
    print('Number of regions: ', num)

    # 7. Watershed Segmentation
    labels = np.int32(labels)
    regions = cv.watershed(imgRGB, labels)
    plt.subplot(337)
    plt.imshow(regions)
    plt.title('Watershed Segmented regions')

    # 8. Marking Watershed labels/regions in the original image
    imgRGB[labels == -1] = [255, 0, 0]                                      # labels == -1 means where black pixel is there, means boundary of regions
    plt.subplot(338)
    plt.imshow(imgRGB) 
    plt.title('Marking Watershed in actual image')

    output_path = os.path.join(root, 'output/25_watershed_segmentation.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()



if __name__ == "__main__":
    watershedSegmentation()
import os 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def graphCutSeg():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure(figsize =  (12,6))
    
    plt.subplot(231)
    plt.imshow(imgRGB)
    plt.title('Original image')

    rows, cols, _ = img.shape
    mask = np.zeros((rows, cols), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    x0 = 30
    y0 = 205
    x1 = 520
    y1 = 360
    rect =  (x0, y0, x1-x0, y1-y0)
    iter = 1

    cv.grabCut(img, mask, rect, bgdModel, fgdModel, iter, cv.GC_INIT_WITH_RECT)

    plt.subplot(232)
    plt.imshow(mask)
    plt.title('Mask')

    # Applying this mask to the image for segmentation
    maskGC = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    imgSeg = img * maskGC[:,:,np.newaxis]
    plt.subplot(233)
    plt.imshow(imgSeg)
    plt.title('Segmented image')


    # Manually filling the gaps area in the foreground object
    maskPath = os.path.join(root, 'data/xyz.jpg')
    markedMask = cv.imread(maskPath, cv.IMREAD_GRAYSCALE)
    mask[markedMask == 255] = 1

    cv.grabCut(img, mask, None, bgdModel, fgdModel, iter, cv.GC_INIT_WITH_MASK)
    plt.subplot(235)
    plt.imshow(mask)
    plt.title('Marked mask')

    maskGC = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    imgSeg = img * maskGC[:,:,np.newaxis]
    plt.subplot(236)
    plt.imshow(imgSeg)
    plt.title('Segmented image with marked mask')


    output_path = os.path.join(root, 'output/26_grabCutSegmentation.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()
     


if __name__ == "__main__":
    graphCutSeg()


# NOTE: Here, there is no need for manually filling the segmented image, as my segmentation is correct. 
# But this method can be used for filling some cavities in the foreground object image.
# The only issue is I dont know what is markedMask image is and what's its use ?
# One disadvantage of this manual filling method is that, it can lead to create gap area in other parts of the foreground.
# NOTE: This is not implemented as SURF is patented algorithm in OpenCV

import os 
import cv2 as cv
import matplotlib.pyplot as plt



def SURF():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    plt.figure(figsize = (12,6))
    plt.subplot(121)
    plt.imshow(imgRGB)
    plt.title('Original')

    hessianThreshold = 3000
    surf = cv.xfeatures2d.SURF_create(hessianThreshold)

    keypoints = surf.detect(imgGRAY, None)

    print('No. of Keypoints: ', len(keypoints))

    print('\nKeypoints: ', keypoints)

    imgGRAY = cv.drawKeypoints (imgGRAY, keypoints, imgGRAY, flags= cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.subplot(122)
    plt.imshow(imgRGB)
    plt.title('SIFT Feature detection')

    output_path = os.path.join(root, 'output/29_SURF_feature_det.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()




if __name__ == "__main__":
    SURF()


'''
# NOTE:

- xfeatures2d belongs to OpenCV's contrib/extra modules, not the normal opencv-python package.

- Also, SURF is a patented/non-free algorithm in OpenCV, so depending on your OpenCV version/build, it may not be available even with contrib installed.

'''
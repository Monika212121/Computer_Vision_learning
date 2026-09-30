import os 
import cv2 as cv
import matplotlib.pyplot as plt



def FAST():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    imgGRAY = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    fast = cv.FastFeatureDetector_create()
    minIntensityDiff = 90

    fast.setThreshold(minIntensityDiff)

    keypoints = fast.detect(imgGRAY)
    imgGRAY = cv.drawKeypoints(imgGRAY, keypoints, imgGRAY, flags= cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(imgGRAY)
    plt.title('Fast Corner detection')

    output_path = os.path.join(root, 'output/31_FAST_corner_detection.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()




if __name__ == "__main__":
    FAST()
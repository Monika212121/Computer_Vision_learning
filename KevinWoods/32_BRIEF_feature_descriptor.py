# NOTE: This is not implemented similar to the SURF , due to OpenCV's old xfeatures2d module and i am using newer version of OpenCV.
import os 
import cv2 as cv




def BRIEF():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    imgGRAY = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)\

    fast = cv.FastFeatureDetector_create()

    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    keypoints = fast.detect(imgGRAY, None)

    keypoints, descriptors = brief.compute(imgGRAY, keypoints)

    print(brief.descriptorSize())

    print(descriptors[0])

    print(' '.join([format(val, '08b') for val in descriptors[0]]))



if __name__ == "__main__":
    BRIEF()
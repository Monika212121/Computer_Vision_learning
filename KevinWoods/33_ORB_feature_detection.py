import os 
import cv2 as cv
import matplotlib.pyplot as plt



def ORB():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    imgGRAY = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    orb = cv.ORB_create()
    keypoints = orb.detect(imgGRAY, None)

    keypoints, _ = orb.compute(imgGRAY, keypoints)
    imgGRAY = cv.drawKeypoints(imgGRAY, keypoints, imgGRAY, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(imgGRAY)
    plt.title('ORB Feature detection')

    output_path = os.path.join(root, 'output/33_ORB_feature_detection.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')


    plt.show()


if __name__ == "__main__":
    ORB()
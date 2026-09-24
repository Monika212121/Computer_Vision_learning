import os 
import cv2 as cv
import matplotlib.pyplot as plt



def SIFT():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    plt.figure(figsize = (12,6))
    plt.subplot(121)
    plt.imshow(imgRGB)
    plt.title('Original')

    sift = cv.SIFT_create()
    keypoints = sift.detect(imgGRAY, None)

    print('No. of Keypoints: ', len(keypoints))

    print('\nKeypoints: ', keypoints)

    imgGRAY = cv.drawKeypoints(imgGRAY, keypoints, imgRGB, flags= cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.subplot(122)
    plt.imshow(imgRGB)
    plt.title('SIFT Feature detection')

    output_path = os.path.join(root, 'output/29_SIFT_feature_det.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()




if __name__ == "__main__":
    SIFT()
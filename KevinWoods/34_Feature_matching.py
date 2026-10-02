import os 
import cv2 as cv
import matplotlib.pyplot as plt



def bruteForce():
    root = os.getcwd()
    imgPath1 = os.path.join(root, 'data/bugatti_cheron_angle1.jpg')
    imgPath2 = os.path.join(root, 'data/bugatti_cheron_ang3.jpg')

    imgGRAY1 = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
    imgGRAY2 = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)

    orb = cv.ORB_create()
    keypoints1, descriptor1 = orb.detectAndCompute(imgGRAY1, None)
    keypoints2, descriptor2 = orb.detectAndCompute(imgGRAY2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck= True)

    matches = bf.match(descriptor1, descriptor2)
    matches = sorted(matches, key = lambda x: x.distance)
    nMatches = 20
    imgMatch = cv.drawMatches(imgGRAY1, keypoints1, imgGRAY2, keypoints2, matches[:nMatches], None, flags= cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure()
    plt.imshow(imgMatch)
    plt.title('Brute Force Feature Matching')

    output_path = os.path.join(root, 'output/34_1_brute_feature_det.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()



def knnBruteForce():
    root = os.getcwd()
    imgPath1 = os.path.join(root, 'data/bugatti_cheron_angle1.jpg')
    imgPath2 = os.path.join(root, 'data/bugatti_cheron_ang3.jpg')

    imgGRAY1 = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
    imgGRAY2 = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(imgGRAY1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(imgGRAY2, None)

    bf = cv.BFMatcher()
    nNeighbors = 2
    matches = bf.knnMatch(descriptor1, descriptor2, nNeighbors)

    goodMatches = []
    testRatio = 0.75

    for m, n in matches:
        if m.distance < testRatio * n.distance:
            goodMatches.append([m])


    imgMatch = cv.drawMatchesKnn(imgGRAY1, keypoints1, imgGRAY2, keypoints2, goodMatches, None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure()
    plt.imshow(imgMatch)
    plt.title('KNN Brute Force Feature Matching')

    output_path = os.path.join(root, 'output/34_2_KNN_brute_feature_det.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()




def FLANN():
    root = os.getcwd()
    imgPath1 = os.path.join(root, 'data/bugatti_cheron_angle1.jpg')
    imgPath2 = os.path.join(root, 'data/bugatti_cheron_ang3.jpg')

    imgGRAY1 = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
    imgGRAY2 = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(imgGRAY1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(imgGRAY2, None)

    FLANN_INDEX_KDTREE = 1
    nKDtrees = 5
    nLeafChecks = 50
    nNeighbors = 2
    indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = nKDtrees)
    searchParams = dict(checks = nLeafChecks)

    flann = cv.FlannBasedMatcher(indexParams, searchParams)

    matches = flann.knnMatch(descriptor1, descriptor2, k= nNeighbors)
    matchesMask = [[0,0] for i in range(len(matches))]
    testRatio = 0.75

    for i, (m,n) in enumerate(matches):
        if m.distance < testRatio * n.distance:
            matchesMask[i] = [1,0]

    drawParams = dict(matchColor = (0, 255, 0), singlePointColor= (255,0,0), matchesMask = matchesMask, flags = cv.DrawMatchesFlags_DEFAULT)
    imgMatch = cv.drawMatchesKnn(imgGRAY1, keypoints1, imgGRAY2, keypoints2, matches, None, **drawParams)

    plt.figure()
    plt.imshow(imgMatch)
    plt.title('FLANN Feature Matching')

    output_path = os.path.join(root, 'output/34_3_flann_feature_det.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')


    plt.show()




if __name__ == "__main__":
    bruteForce()
    knnBruteForce()
    FLANN()
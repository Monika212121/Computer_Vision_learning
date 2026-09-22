import os 
import cv2 as cv
import matplotlib.pyplot as plt



def goodCornerDetection():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    maxCorners = 75
    quality = 0.01
    minDistance = 30

    corners = cv.goodFeaturesToTrack(imgGRAY, maxCorners, quality, minDistance)
    print("No. of corners: ", len(corners))
    print("\ncorners: \n", corners)

    for corner in corners:
        x = int(corner[0][0])
        y = int(corner[0][1])
        cv.circle(imgRGB, (x,y), 2, (255, 0, 0), -1)

    plt.figure()
    plt.imshow(imgRGB)
    plt.title('Good Corners marked')


    output_path = os.path.join(root, 'output/28_good_corner.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()




if __name__ == "__main__":
    goodCornerDetection()

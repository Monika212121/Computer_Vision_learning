# NOTE: Our main aim here is to identify circles in the image and extract them.

import os
import cv2 as cv
import matplotlib.pyplot as plt



def houghCircleTransform():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car2.webp')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    imgGRAY = cv.medianBlur(imgGRAY, 21)

    circles = cv.HoughCircles(imgGRAY, cv.HOUGH_GRADIENT, dp= 1, minDist= 300, param1= 200, param2= 10, minRadius= 75, maxRadius= 150)

    print("circles: ", circles)

    print("\ncircles[0,:]", circles[0,:])

    for circle in circles[0,:]:
        x_centre, y_center = int(circle[0]), int(circle[1])
        radius = int(circle[2])

        cv.circle(imgRGB, (x_centre, y_center), radius, (255, 0, 0), 10)


    plt.figure()
    plt.imshow(imgRGB)
  
    output_path = os.path.join(root, 'output/24_1_car_circle_hough_trans.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()




if __name__ == "__main__":
    houghCircleTransform()
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def contours():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/tesla_logo.webp')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    # 1. Taking only logo region
    img = img[206:421, 380:660]

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap = 'gray')
    plt.title('Original Grayscale')
    
    h, w = img.shape
    scale = 4
    new_h, new_w = int(scale*h), int(scale*w)
    img = cv.resize(img, (new_w, new_h))

    # Draw Histogram of logo, to find it's Threshold value
    #hist = cv.calcHist([img], [0], None, [256], [0,256])
    #plt.subplot(232)
    #plt.plot(hist)
    
    # 2. Binary mask
    _, imgThresh = cv.threshold(img, 205, 255, cv.THRESH_BINARY)
    kernel = np.ones((7,7), np.uint8)

    # To enlarge boundaries of objects in the image
    imgThresh = cv.dilate(imgThresh, kernel)                                            # to show image details better

    plt.subplot(232)
    plt.imshow(imgThresh, cmap = 'gray')
    plt.title('Binary mask(dilated)')

    # 3. Implmenting contours to find all possible contours in the image
    contours, _ = cv.findContours(imgThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)      # RETR TREE will get all contours without heirarchy

    # Taking the first and biggest contour(relevant)
    contours = [contours[0]]
    cv.drawContours(img, contours, -1, (0, 255, 0), 5)                                  # drawing contours n the image(in reality, I can't see anthing)

    plt.subplot(233)
    plt.imshow(img, cmap = 'gray')
    plt.title('Contours')

    # 4. Finding COM(Centre Of Mass) of the image
    M = cv.moments(contours[0])
    Cx = int(M['m10']/M['m00'])
    Cy = int(M['m01']/M['m00'])

    plt.subplot(234)
    plt.imshow(img, cmap = 'gray')
    plt.plot(Cx, Cy, 'r*')
    plt.title('Center Of Mass')

    # Finding Area and Perimeter of contour
    area = cv.contourArea(contours[0])
    perimeter = cv.arcLength(contours[0], True)

    print("\nArea of contour: ", area)
    print("\nPerimeter of contour: ", perimeter)

    # Contour Approximation
    epsilon = .01 * perimeter
    approx = cv.approxPolyDP(contours[0], epsilon, True)
    approx = np.array(approx)
    approx = np.concatenate((approx, approx[:1]), axis= 0)

    plt.plot(approx[:, 0, 0], approx[:, 0, 1])

    # 5. Draw boundary with contour edge points in image
    hull = cv.convexHull(contours[0])
    hull = hull[:, 0, :]
    hull = np.concatenate((hull, hull[:1]), axis = 0)

    plt.subplot(235)
    plt.imshow(img, cmap= 'gray')
    plt.plot(hull[:,0], hull[:,1], 'r-')
    plt.title('Hull')

    # 6. Draw rectangle Bounding Box around an object in image
    x, y, w, h = cv.boundingRect(contours[0])

    plt.subplot(236)
    cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)
    plt.imshow(img, cmap= 'gray')
    plt.title('Bounding box')

    output_path = os.path.join(root, 'output/20_contours.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()


    # Some useful contour operations
    aspect_ratio = w/h
    extent = w*h
    solidity = area/cv.contourArea(hull)
    equiDIa = np.sqrt(4*area/np.pi)
    _, _, angle = cv.fitEllipse(contours[0])                                            # to find alignment of an object in image

    print("\nAspect ratio: ", aspect_ratio)
    print("\nExtent: ", extent)
    print("\nSolidity: ", solidity)
    print("\nequiDIa: ", equiDIa)
    print("\nAngle: ", angle)



if __name__ == "__main__":
    contours()


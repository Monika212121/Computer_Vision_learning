import os
import cv2 as cv
import matplotlib.pyplot as plt



def templateMatching():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car2.webp')
    img = cv.imread(imgPath)

    logo = img[510:550, 370:425]
    height, width, _ = logo.shape

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Original image')

    plt.subplot(122)
    plt.imshow(logo)
    plt.title('Template')

    output_path = os.path.join(root, 'output/22_1_imageLogo.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR, cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]
    titles = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']


    plt.figure(figsize = (12,12))

    for i in range(len(methods)):
        curImg = img.copy()

        templateMap = cv.matchTemplate(curImg, logo, methods[i])
        _,_, minLoc, maxLoc = cv.minMaxLoc(templateMap)
        
        print(f"\nmethod: {titles[i]}\n")
        print(f"Template Map size: {templateMap.size}\n")
        print("templateMap: \n", templateMap)
        print(f"minLoc: {minLoc}, maxLoc: {maxLoc}\n")

        if methods[i] == cv.TM_SQDIFF or methods[i] == cv.TM_SQDIFF_NORMED:
            topLeft = minLoc                                                        # corrosponds to Differnece / Error
        else:
            topLeft = maxLoc                                                        # corrosponds to Matching score

        bottomRight = (topLeft[0] + width, topLeft[1] + height)

        cv.rectangle(curImg, topLeft, bottomRight, (255,0,0), 20)           

        plt.subplot(6, 3, 3*i + 1)
        plt.text(0.5, 0.5, titles[i], ha = 'center', va = 'center', fontsize = 10)
        plt.axis('off')

        plt.subplot(6, 3, 3*i + 2)
        plt.imshow(templateMap)

        plt.subplot(6, 3, 3*i + 3)
        plt.imshow(curImg)


    output_path = os.path.join(root, 'output/22_2_templateMatching.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()




if __name__ == "__main__":
    templateMatching()



# NOTE: Only "TM_CCORR" method gave wrong template matching, otherwise we get accurate logo positions in the original image.
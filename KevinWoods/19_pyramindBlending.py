import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def pyramindBlending():

    root = os.getcwd()
    imgPath = os.path.join(root, 'data', 'car.png')

    imgBGR = cv.imread(imgPath)

    if imgBGR is None:
        raise FileNotFoundError(f"Could not read image: {imgPath}")

    imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB)


    # ============================================================
    # BGR PYRAMIDS
    # ============================================================

    plt.figure(figsize=(12, 6))

    downSamp = imgBGR.copy()
    BGR_gausPyramidList = [downSamp]

    plt.subplot(2, 3, 1)
    plt.imshow(cv.cvtColor(downSamp, cv.COLOR_BGR2RGB))
    plt.title("BGR Gaussian G0")

    for i in range(5):

        downSamp = cv.pyrDown(downSamp)
        BGR_gausPyramidList.append(downSamp)

        plt.subplot(2, 3, i + 2)
        plt.imshow(cv.cvtColor(downSamp, cv.COLOR_BGR2RGB))
        plt.title(f"BGR Gaussian G{i + 1}")
        

    output_path = os.path.join(root, 'output/19_1_BGR_Gaussian.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    # ============================================================
    # BGR LAPLACIAN PYRAMID
    # ============================================================

    plt.figure(figsize=(12, 6))

    # [G5, L4, L3, L2, L1]
    BGR_lapPyramidList = [BGR_gausPyramidList[-1]]

    for i in range(len(BGR_gausPyramidList) - 1, 0, -1):

        currentGaussian = BGR_gausPyramidList[i - 1]
        smallerGaussian = BGR_gausPyramidList[i]

        upSamp = cv.pyrUp(
            smallerGaussian,
            dstsize=(
                currentGaussian.shape[1],
                currentGaussian.shape[0]
            )
        )

        # Use float32 because Laplacian values can be negative.
        diff = (
            currentGaussian.astype(np.float32)
            - upSamp.astype(np.float32)
        )

        BGR_lapPyramidList.append(diff)

        plt.subplot(2, 3, len(BGR_lapPyramidList) - 1)
        plt.imshow(cv.cvtColor(np.clip(diff, 0, 255).astype(np.uint8), cv.COLOR_BGR2RGB))
        plt.title(f"BGR Laplacian L{i - 1}")


    output_path = os.path.join(root, 'output/19_2_BGR_Laplacian.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')


    # ============================================================
    # RGB PYRAMIDS
    # ============================================================

    plt.figure(figsize=(12, 6))

    downSamp = imgRGB.copy()
    RGB_gausPyramidList = [downSamp]

    plt.subplot(2, 3, 1)
    plt.imshow(downSamp)
    plt.title("RGB Gaussian G0")

    for i in range(5):

        downSamp = cv.pyrDown(downSamp)
        RGB_gausPyramidList.append(downSamp)

        plt.subplot(2, 3, i + 2)
        plt.imshow(downSamp)
        plt.title(f"RGB Gaussian G{i + 1}")


    output_path = os.path.join(root, 'output/19_3_RGB_Gaussian.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')


    # ============================================================
    # RGB LAPLACIAN PYRAMID
    # ============================================================

    plt.figure(figsize=(12, 6))

    # [G5, L4, L3, L2, L1]
    RGB_lapPyramidList = [RGB_gausPyramidList[-1]]

    for i in range(len(RGB_gausPyramidList) - 1, 0, -1):

        currentGaussian = RGB_gausPyramidList[i - 1]
        smallerGaussian = RGB_gausPyramidList[i]

        upSamp = cv.pyrUp(
            smallerGaussian,
            dstsize=(
                currentGaussian.shape[1],
                currentGaussian.shape[0]
            )
        )

        # Use float32 because Laplacian values can be negative.
        diff = (
            currentGaussian.astype(np.float32)
            - upSamp.astype(np.float32)
        )

        RGB_lapPyramidList.append(diff)

        plt.subplot(2, 3, len(RGB_lapPyramidList) - 1)
        plt.imshow(np.clip(diff, 0, 255).astype(np.uint8))
        plt.title(f"RGB Laplacian L{i - 1}")


    output_path = os.path.join(root, 'output/19_4_RGB_Laplacian.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')


    # ============================================================
    # BLEND PYRAMIDS
    # ============================================================

    plt.figure(figsize=(12, 6))

    combinedList = []

    offset = 7

    for i in range(len(RGB_lapPyramidList)):

        left = RGB_lapPyramidList[i]
        right = BGR_lapPyramidList[i]

        cols = left.shape[1]
        split = cols // 2 + offset

        combined = np.hstack((
            left[:, :split],
            right[:, split:]
        ))

        combinedList.append(combined)

        plt.subplot(2, 3, i + 1)
        plt.imshow(np.clip(combined, 0, 255).astype(np.uint8))
        plt.title(f"Combined Level {i}")


    output_path = os.path.join(root, 'output/19_5_BGR_Gaussian.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')


    # ============================================================
    # RECONSTRUCT BLENDED IMAGE
    # ============================================================

    blend = combinedList[0]

    # [G5, L4, L3, L2, L1]
    for i in range(1, len(combinedList)):

        blend = cv.pyrUp(blend, dstsize=(combinedList[i].shape[1], combinedList[i].shape[0]))

        blend = blend + combinedList[i]


    # Convert reconstructed image back to uint8.
    blend = np.clip(blend, 0, 255).astype(np.uint8)


    # ============================================================
    # DISPLAY FINAL RESULT
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.imshow(blend)
    plt.title("Pyramid Blended Image")

    output_path = os.path.join(root, 'output/19_6_blendedImage.jpg')
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()



if __name__ == "__main__":
    pyramindBlending()
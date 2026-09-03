import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def meanFiltering():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    imgRBG = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    h, w, _ = img.shape
    print(f"Image shape: ", imgRBG.shape)

    # -------------------------------ADDING SALT AND PEPPER NOISE-----------------------------------

    # Make a copy so that the original image remains unchanged.
    noisyImg = imgRBG.copy()

    # Total probability of adding noise to a pixel.
    # 0.05 = approximately 5% of pixels will be corrupted.
    noiseProb = 0.05

    # Generate one random float number between 0 and 1 for every pixel.
    #
    # Shape: (h, w)
    # dtype: float64
    #
    # Each value acts like a random decision for that pixel:
    #   < 0.025       -> black noise
    #   > 0.975       -> white noise
    #   otherwise     -> leave pixel unchanged
    noiseRandom = np.random.rand(h, w)
    print("NoiseRandom shape: ", noiseRandom.shape)
    print("\nNoiseRandom: \n", noiseRandom)


    # ---------------------------------------------------------
    # BLACK / "PEPPER" NOISE
    # ---------------------------------------------------------
    # Create a Boolean mask:
    # True wherever the random value is < 0.025.
    # Since 0.025 = noiseProb / 2, approximately 2.5% of pixels will be selected.
    blackMask = noiseRandom < noiseProb / 2

    # Boolean indexing:
    # Select all pixels where blackMask is True and set their RGB values to [0, 0, 0] (black).
    noisyImg[blackMask] = 0


    # ---------------------------------------------------------
    # WHITE / "SALT" NOISE
    # ---------------------------------------------------------
    # Select approximately another 2.5% of pixels.
    whiteMask = noiseRandom > 1 - noiseProb / 2

    # Set selected RGB pixels to [255, 255, 255] (white).
    noisyImg[whiteMask] = 255


    # ---------------------------------------------------------
    # MEDIAN FILTER
    # ---------------------------------------------------------
    # Apply a 5x5 median filter/Kernel to remove salt-and-pepper noise.
    # For every pixel, the filter looks at its 5x5 neighborhood, sorts the values, and replaces the center pixel with the median value.
    # Median filtering is particularly effective against "Salt-and-pepper" noise because extreme values like 0 and 255 are treated as outliers.

    imgFilter = cv.medianBlur(noisyImg, 5)


    plt.figure(figsize = (12, 6))
    plt.subplot(131)
    plt.imshow(imgRBG)
    plt.title('Original image')

    plt.subplot(132)
    plt.imshow(noisyImg)
    plt.title('Noisy image')

    plt.subplot(133)
    plt.imshow(imgFilter)
    plt.title('Blurred/ Smoothed image')


    output_path = os.path.join(root, 'output/09_MeanFiltering.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()



if __name__ == "__main__":
    meanFiltering()



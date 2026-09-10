import os
import cv2 as cv
import matplotlib.pyplot as plt



def imageGradient():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    imgGRAY = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    plt.figure(figsize = (12,6))

    plt.subplot(221)
    plt.imshow(imgGRAY, cmap='gray')
    plt.title('Original Grayscale')

    # 2nd Order derivative
    laplacian = cv.Laplacian(imgGRAY, cv.CV_64F, ksize= 21)
    plt.subplot(222)
    plt.imshow(laplacian, cmap = 'gray')
    plt.title('Laplacian')


    # 1st Order derivatives

    # 1. Sobel X
    kx, ky = cv.getDerivKernels(1, 0 ,3)
    print("******************** SobelX ********************")
    print("\nkx: ", kx, "\nshape: ", kx.shape)
    print("\nky: \n", ky, "\nshape: ", ky.shape)
    print("\nky@kx.T: \n", ky@kx.T)                                     # Sobel X kernel

    solbelX = cv.Sobel(imgGRAY, cv.CV_64F, 1, 0, ksize= 21)
    plt.subplot(223)
    plt.imshow(solbelX, cmap= 'gray') 
    plt.title('Sobel X')

    # 2. Sobel Y
    kx, ky = cv.getDerivKernels(0, 1 ,3)
    print("\n******************** SobelY ********************")
    print("\nkx: ", kx, "\nshape: ", kx.shape)
    print("\nky: \n", ky, "\nshape: ", ky.shape)
    print("\nky@kx.T: \n", ky@kx.T)                                     # Sobel Y kernel

    solbelY = cv.Sobel(imgGRAY, cv.CV_64F, 0, 1, ksize= 21)
    plt.subplot(224)
    plt.imshow(solbelY, cmap= 'gray')
    plt.title('Sobel Y')

    output_path = os.path.join(root, 'output/17_imageGradient.jpg') 
    plt.savefig(output_path, bbox_inches = 'tight')

    plt.show()





if __name__ == "__main__":
    imageGradient()
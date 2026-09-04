import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def createGaussianKernel(kernel_size: int, sigma: int):
    kernel = cv.getGaussianKernel(ksize= kernel_size, sigma= sigma)
    kernel = np.outer(kernel, kernel)
    return kernel


def callback(input):
    pass


def gaussianFiltering():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data/car.png')
    img = cv.imread(imgPath)

    n = 101
    kernel = createGaussianKernel(kernel_size = n, sigma = 8)

    fig = plt.figure()

    # Visualizing Kernel in 1D and 2D
    plt.subplot(121)
    plt.imshow(kernel)
    plt.title(f'Kernel of size = {n}')

    ax = fig.add_subplot(122, projection='3d')
    x = np.arange(0, n, 1)
    y = np.arange(0, n, 1)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, kernel, cmap = 'viridis')
    plt.title('Kernel in 2D')

    kernel_output_path = os.path.join(root, 'output/10_1_gaussianKernel.jpg') 
    plt.savefig(kernel_output_path, bbox_inches = 'tight')

    plt.show()

    # Creating trackbar to visualize blurring with dynamic "Sigma" value
    winName = "Gaussian Filtering"
    cv.namedWindow(winName)
    cv.createTrackbar('sigma', winName, 1, 20, callback)


    while True:
        if cv.waitKey(1) == ord('q'):
            break

        # Getting sigma value from trackbar
        sigma = cv.getTrackbarPos('sigma', winName)

        # Gaussian Filtering according to the dynamic sigma value
        imgFilter = cv.GaussianBlur(img, (n,n), sigma)
        cv.imshow(winName, imgFilter)

        output_path = os.path.join(root, 'output/10_2_gaussianFiltering.jpg') 
        cv.imwrite(output_path, imgFilter)


    cv.destroyAllWindows()



if __name__ == "__main__":
    gaussianFiltering()



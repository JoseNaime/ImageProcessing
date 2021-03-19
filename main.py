import numpy as np
import cv2 as cv


# Kernel Filters
class KERNEL_FILTERS():
    BORDERS = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])
    SOBEL_X = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    SOBEL_Y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])


# Load image in Grayscale
def loadImage(url):
    return cv.imread(url, cv.IMREAD_GRAYSCALE)


# Applies kernel filter to image section
def getKernel(section, kernel):
    n = section.size
    return np.sum((section * kernel)) / n


# Add Kernel filter and image padding
def convolution(img, kernel, padding=1):
    iRows, iColumns = img.shape
    kRows, kColumns = kernel.shape

    # Calculate padding [(W−K+2P)/S]+1
    iColumnsFiltered = int((iColumns - kColumns + 2 * padding) + 1)
    iRowsFiltered = int((iRows - kRows + 2 * padding) + 1)
    response = np.zeros((iRowsFiltered, iColumnsFiltered))

    paddingImg = img
    if padding != 0:
        paddingImg = np.zeros(
            (iRowsFiltered + 2 * padding, iColumnsFiltered + 2 * padding))
        paddingImg[padding:-padding, padding:-padding] = img

    for i in range(iRowsFiltered - kRows):
        for j in range(iColumnsFiltered - kColumns):
            response[i, j] = getKernel(
                paddingImg[i:i + kRows,
                j:j + kColumns], kernel)
    return response


# Main Code
if __name__ == '__main__':
    img = loadImage('./images/Image1.png')

    cv.imshow("Original", img)
    cv.imshow("Laplacian", convolution(img, KERNEL_FILTERS.BORDERS))
    cv.imshow("Sobel X", convolution(img, KERNEL_FILTERS.SOBEL_X))
    cv.imshow("Sobel Y", convolution(img, KERNEL_FILTERS.SOBEL_Y))

    cv.waitKey(0)
    cv.destroyAllWindows()

import numpy as np
import cv2 as cv

# Asigna filtros kernel
BORDERS = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])


# Carga la imagen en Grayscale
def loadImage(url):
    return cv.imread(url, cv.IMREAD_GRAYSCALE)


# Multiplica la seccion por los valores Kernel
def getKernel(section, kernel):
    n = section.size
    return np.sum((section * kernel)) / n


def convolution(img, kernel, padding=1):
    iRows, iColumns = img.shape
    kRows, kColumns = kernel.shape

    # [(W−K+2P)/S]+1
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


if __name__ == '__main__':
    img = loadImage('./images/Image1.png')

    cv.imshow("Original", img)
    cv.imshow("Laplacian", convolution(img, BORDERS))
    cv.imshow("Sobel X", convolution(img, SOBEL_X))
    cv.imshow("Sobel Y", convolution(img, SOBEL_Y))
    cv.waitKey(0)
    cv.destroyAllWindows()

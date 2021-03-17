from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv


def loadImage(url):
    return cv.imread(url, cv.IMREAD_GRAYSCALE)


img = loadImage('./Image1.png')
cv.imshow("Original", img)

borders = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

sobelX = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

sobelY = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])


def getKernel(section, kernel):
    rows, cols = section.shape
    result = 0.0
    for i in range(rows):
        for j in range(cols):
            result += section[i, j] * kernel[i, j]
    return result


def convolution(img, kernel):
    iRows, iColumns = img.shape
    kRows, kColumns = kernel.shape

    response = np.zeros(img.shape)  # matriz donde guardo el resultado

    for i in range(iRows):
        for j in range(iColumns):
            response[i, j] = getKernel(
                img[i:i + kRows,
                j:j + kColumns], kernel)
    return response


cv.imshow("Laplacian", convolution(img, borders))
cv.imshow("Sobel X", convolution(img, sobelX))
cv.imshow("Sobel Y", convolution(img, sobelY))
cv.waitKey(0)
cv.destroyAllWindows()

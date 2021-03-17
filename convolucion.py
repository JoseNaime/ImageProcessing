# %%
import numpy as np
import cv2 as cv


# %% Load Image as Grayscale
def loadImage(url):
    return cv.imread(url, cv.IMREAD_GRAYSCALE)

originalImg = loadImage('./Image.png')

cv.imshow('image', originalImg)

# %%
def getKernel(img, scale):
    mul = np.matrix([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    res = img.copy()
    rows, columns = img.shape
    for i in range(rows - (1 + scale)):
        for j in range(columns - (1 + scale)):
            res[i + scale][j + scale] = np.sum(img[i:scale + i, j:scale + j] * mul) / 9
    return res


kernel = getKernel(originalImg, 3)


cv.imshow('Kernel', kernel)
cv.waitKey(0)

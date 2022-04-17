# ---------------------------------
# CS472 - Assignment 3
# Date: 16/04/2022
# Name Surname: Adel Abbas
# Registration number: csdp1270
#
# --------------------------------

import os
import cv2 as cv
import numpy as np
import math


def readImage(filename):
    return cv.imread(filename, cv.IMREAD_GRAYSCALE)


def writeImage(img, filename):
    cv.imwrite(filename, img)


def processImage(filename, data="./", out="./"):
    print("-------------------------")
    print(f"Processing {filename}...")
    if filename.endswith("jpg"):
        img0 = readImage(os.path.join(data, filename))
        writeImage(img0, os.path.join(out, filename+"_greyscale.jpg"))
        print(" Applying Gaussian smoothing")
        h = getGaussianKernel(1)
        img1 = myImageFilter(img0, h)
        writeImage(img1, os.path.join(out, filename+"_smoothed.jpg"))

        print(" Applying Sobel filters")
        sobelx = getSobelHorizontalKernel()
        sobely = getSobelVerticalKernel()
        gx = myImageFilter(img1, sobelx)
        writeImage(gx, os.path.join(out, filename+"_sobelx.jpg"))
        gy = myImageFilter(img1, sobely)
        writeImage(gy, os.path.join(out, filename+"_sobely.jpg"))

        print(" Computing gradient magnitude")
        gradient, orientation = getGradient(img1)
        writeImage(gradient, os.path.join(out, filename+"_gradient.jpg"))

        print(" Applying naïve non-maxima suppression")
        nms = naiveNonMaximaSuppression(gradient)
        writeImage(nms, os.path.join(out, filename+"_nms.jpg"))

        print(" Applying discrete non-maxima suppression")
        dnms = discreteNonMaximaSuppression(gradient, orientation)
        writeImage(dnms, os.path.join(out, filename+"_nms2.jpg"))


"""
Computes the gradient magnitude and orientation for the input image.
It first applies Gaussian smoothing with the provided standard deviation sigma.
Then computes gradient magnitude and orientation, and finally a non-maxima 
suppression.
"""


def myEdgeFilter(img, sigma):
    h = getGaussianKernel(sigma)
    img1 = myImageFilter(img, h)
    gradient, orientation = getGradient(img1)
    dnms = discreteNonMaximaSuppression(gradient, orientation)
    return dnms

"""
Adds zero padding to the input image
"""


def zeroPadImage(img, rowPadding, colPadding):
    shape = np.shape(img)
    rows = shape[0] + 2 * rowPadding
    cols = shape[1] + 2 * colPadding
    paddedImg = np.zeros((rows, cols))
    paddedImg[rowPadding: rowPadding + shape[0],
              colPadding: colPadding + shape[1]] = img
    return paddedImg


"""
Adds padding to the image by filling the padded regions with the values of
the closest image pixels.
"""


def padImage(img, rowPadding, colPadding):
    shape = np.shape(img)
    rows = shape[0] + 2 * rowPadding  # pad on top and bottom
    cols = shape[1] + 2 * colPadding  # pad on the left and on the right

    paddedImg = np.zeros((rows, cols))

    # Compute padding values
    top = np.tile(img[0, :], (rowPadding, 1))
    bottom = np.tile(img[-1, :], (rowPadding, 1))
    left = np.tile(img[:, 0], (colPadding, 1)).T
    right = np.tile(img[:, -1], (colPadding, 1)).T
    top_left = np.ones((rowPadding, colPadding)) * img[0][0]
    top_right = np.ones((rowPadding, colPadding)) * img[0][-1]
    bottom_left = np.ones((rowPadding, colPadding)) * img[-1][0]
    bottom_right = np.ones((rowPadding, colPadding)) * img[-1][-1]

    # top
    paddedImg[:rowPadding, :colPadding] = top_left
    paddedImg[:rowPadding, colPadding:colPadding + shape[1]] = top
    paddedImg[:rowPadding, colPadding + shape[1]:] = top_right

    # center
    paddedImg[rowPadding:rowPadding + shape[0], :colPadding] = left
    paddedImg[rowPadding: rowPadding + shape[0],
              colPadding: colPadding + shape[1]] = img
    paddedImg[rowPadding:rowPadding + shape[0], colPadding + shape[1]:] = right

    # bottom
    paddedImg[rowPadding+shape[0]:, :colPadding] = bottom_left
    paddedImg[rowPadding+shape[0]:, colPadding:colPadding + shape[1]] = bottom
    paddedImg[rowPadding+shape[0]:, colPadding + shape[1]:] = bottom_right

    return paddedImg


"""
Performs convolution of the image with the provided kernel h
The image should be a grayscale image (i.e. a 2D numpy array)
and the provided kernel matrix h is assumed to have odd dimensions
in both directions
"""


def myImageFilter(img, h):
    # Padding values depend on the kernel dimensions
    hshape = np.shape(h)
    ishape = np.shape(img)
    rowPadding = hshape[0] // 2
    colPadding = hshape[1] // 2
    padded = padImage(img, rowPadding, colPadding)

    output = np.zeros(ishape)

    for y in range(rowPadding, ishape[0] + rowPadding):
        for x in range(colPadding, ishape[1] + colPadding):
            window = padded[y - rowPadding:y + rowPadding +
                         1, x - colPadding:x + colPadding + 1]
            s = (window * h).sum()
            output[y - rowPadding, x - colPadding] = s
    return output


"""
Computes both the Gradient magnitude and the orientation for the provided greyscale image, 
using 3x3 Sobel vertical and horizontal Kernels.
"""


def getGradient(img):
    sobelx = getSobelHorizontalKernel()
    sobely = getSobelVerticalKernel()
    gx = myImageFilter(img, sobelx)
    gy = myImageFilter(img, sobely)
    gradient = np.sqrt(np.square(gx) + np.square(gy))
    gx[np.where(gx == 0)] = 1 # Approximate zero values to 1 before computing gy/gx
    orientation = np.arctan(gy/gx)
    return gradient, orientation


"""
Computes both the Gradient magnitude for the provided greyscale image, 
using 3x3 Sobel vertical and horizontal kernels.
"""


def getGradientMagnitude(img):
    sobelx = getSobelHorizontalKernel()
    sobely = getSobelVerticalKernel()
    gx = myImageFilter(img, sobelx)
    gy = myImageFilter(img, sobely)
    gradient = np.sqrt(np.square(gx) + np.square(gy))
    return gradient


"""
Compute the gradient orientation for the provided greyscale image,
using 3x3 Sobel vertical and horizontal kernels.
"""


def getGradientOrientation(img):
    sobelx = getSobelHorizontalKernel()
    sobely = getSobelVerticalKernel()
    gx = myImageFilter(img, sobelx)
    gy = myImageFilter(img, sobely)
    gx[np.where(gx == 0)] = 1 # Approximate zero values to 1 before computing gy / gx
    orientation = np.arctan(gy/gx)
    return orientation


"""
Performs a naive non maxima suppression on the provided gradient matrix.
Each gradient magnitude is tested against its 3x3 neighborhood and is removed, 
if it is not the only occurence of the maximum value.
"""


def naiveNonMaximaSuppression(gradient):
    shape = np.shape(gradient)
    pad = 3
    padded = zeroPadImage(gradient, pad, pad)
    output = np.copy(gradient)
    for y in range(pad, shape[0] + pad):
        for x in range(pad, shape[1] + pad):
            roi = padded[y - pad:y + pad + 1, x - pad:x + pad + 1]
            max = np.max(roi)
            # number of occurences of the max value
            count = np.count_nonzero(roi == max)
            if max != padded[y][x] or count > 1:
                output[y-pad][x-pad] = 0

    return output


"""
The input angle value is assumed to be between 0 and 180°
Returns the closest angle among (0, 45, 90, 135)
"""


def getClosestOrientation(angle):
    angles = [0, 45, 90, 135, 180]
    # Find the index of the angle of minimum distance
    id = (np.abs(angles - angle)).argmin()
    return angles[id] % 180  # if the closest is 180 => return 0


"""
Performs non maxima suppression along a direction determined by discrete
maping of the gradient orientation to 0, 45, 90 or 135 degrees
"""


def discreteNonMaximaSuppression(gradient, orientation):
    shape = np.shape(gradient)
    deg = np.degrees(orientation)
    deg = (deg + 180) % 180  # convert the orientation to 0 - 180° scale
    pad = 3
    output = np.copy(gradient)
    gradient = zeroPadImage(gradient, pad, pad)

    for y in range(pad, shape[0] + pad):
        for x in range(pad, shape[1] + pad):
            angle = getClosestOrientation(deg[y-pad][x-pad])
            a = 0
            b = 0
            if angle == 0:
                a = gradient[y][x-1]
                b = gradient[y][x+1]
            elif angle == 90:
                a = gradient[y-1][x]
                b = gradient[y+1][x]
            elif angle == 45:
                a = gradient[y + 1][x - 1]
                b = gradient[y - 1][x + 1]
            elif angle == 135:
                a = gradient[y - 1][x - 1]
                b = gradient[y + 1][x + 1]

            if gradient[y][x] <= a or gradient[y][x] <= b:
                output[y-pad][x-pad] = 0

    return output


"""
Returns the 2D Gaussian Kernel corresponding to the provided standard deviation.
The kernel is of size h x h, with h = 2 * ceil(3 * sigma) + 1
"""


def getGaussianKernel(sigma = 1):
    hsize = 2 * math.ceil(3 * sigma) + 1
    x = np.linspace(- (hsize//2), hsize // 2, hsize)
    xy, yx = np.meshgrid(x, x)  # returns 2 matrices of distances to the
                                # center pixel on the x and y axis
    ii = np.square(xy)
    jj = np.square(yx)
    kernel = np.exp(-0.5 * (ii + jj) / (sigma**2))
    kernel = kernel / (2 * math.pi * sigma**2)
    return kernel


"""
Returns a 2D 3x3 Horizontal Sobel Kernel
"""


def getSobelHorizontalKernel():
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])


"""
Returns a 2D 3x3 Vertical Sobel Kernel
"""


def getSobelVerticalKernel():
    return np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
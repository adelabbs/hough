#!/usr/bin/env python

# ---------------------------------
# CS472 - Assignment 3
# Date: 16/04/2022
# Name Surname: Adel Abbas
# Student id: csdp1270
# --------------------------------

import os
import cv2 as cv
import numpy as np
import math

# -----------------------------------------------
# Functions from Assignment 2
from gradient import discreteNonMaximaSuppression, getGaussianKernel, getGradient, myImageFilter, naiveNonMaximaSuppression
# -----------------------------------------------

DEBUG = False
CANNY = False
RHO_RES = 1
THETA_RES = 1


def readImage(filename):
    return cv.imread(filename, cv.IMREAD_GRAYSCALE)


def writeImage(img, filename):
    cv.imwrite(filename, img)


def rescaleIntensities(img):
    return (img / np.max(img)) * 255


def myHoughTransform(gradientMagnitude, rhoRes=1, thetaRes=1):
    """
    Computes the Hough accumulator array.
    """
    height, width = gradientMagnitude.shape
    rhoMax = np.sqrt(height * height + width * width)
    thetaMax = 360
    nRhos = math.ceil(rhoMax / rhoRes)
    nThetas = math.ceil(thetaMax / thetaRes)
    hough = np.zeros((nRhos, nThetas))

    # Store sin and cos values to avoid computing them multiple times
    thetas = np.deg2rad(np.arange(0, thetaMax, thetaRes))
    sin = np.sin(thetas)
    cos = np.cos(thetas)

    for y in range(height):
        for x in range(width):
            if gradientMagnitude[y][x] > 127:  # select edge pixels
                for i in range(nThetas):
                    rho = math.ceil(x * cos[i] + y * sin[i])
                    if rho >= 0:  # ignore negative rho values
                        j = rho // rhoRes
                        hough[j][i] += 1

    return hough


def myHoughLines(hough, nLines, rhoRes=1, thetaRes=1):
    """
    Returns the top-n set of parameters in the
    accumulator array.
    """
    lines = np.zeros((nLines, 2))
    width = hough.shape[1]
    nms = naiveNonMaximaSuppression(hough)
    # Convert the hough matrix into a 1-D array
    # to perform sorting
    flat = nms.flatten()
    # Sort coefficients in decreasing order
    sorted = flat.argsort()[::-1]
    topN = sorted[:nLines]
    for k, i in enumerate(topN):
        x = i % width
        y = i // width
        lines[k] = [y * rhoRes, x * thetaRes]
    return lines


def findEndPoints(rho, theta, height, width):
    """
    Returns the end points that define the line parameterized by rho and theta
    in the image of dimensions height x width.
    """
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    if theta == 0 or theta == 180:
        x0 = int(rho)
        y0 = 0
        x1 = int(rho)
        y1 = height - 1
    else:
        theta = np.radians(theta)
        m = - math.cos(theta) / math.sin(theta)
        p = rho / math.sin(theta)
        if DEBUG:
            print(f" Cartesian line equation is : y = {m} * x + {p}")
        if m == 0:
            x0 = 0
            y0 = p
            x1 = width - 1
            y1 = p
        else:
            if p < 0 and p < height:
                y0 = 0
                x0 = math.floor((y0 - p) / m)
            elif p < height:
                x0 = 0
                y0 = math.floor(p)
            else:
                y0 = height - 1
                x0 = math.floor((y0 - p) / m)

            x1 = width - 1
            y1 = math.floor(m * x1 + p)
            if y1 >= height:
                y1 = height - 1
                x1 = math.floor((y1 - p) / m)
            elif y1 < 0:
                y1 = 0
                x1 = math.floor((y1 - p) / m)

    return (x0, y0), (x1, y1)


def drawLines(gradient, lines):
    height, width = gradient.shape
    output = np.dstack([gradient, gradient, gradient])
    for rho, theta in lines:
        if DEBUG:
            print(f" Finding end points for line (rho={rho}, theta={theta})")
        A, B = findEndPoints(rho, theta, height, width)
        if DEBUG:
            print(f" Found {A} and {B}")
        output = cv.line(output, A, B, (0, 0, 255))

    return output


def processImage(filename, data="./", out="./"):
    print("-------------------------")
    print(f"Processing {filename}...")
    if filename.endswith("jpg"):
        img0 = readImage(os.path.join(data, filename))
        filename = os.path.splitext(filename)[0]
        print(" Computing gradient magnitude")
        if CANNY:
            gradient = cv.Canny(image=img0, threshold1=100, threshold2=200)
        else:
            sigma = 1
            h = getGaussianKernel(sigma)
            img = myImageFilter(img0, h)
            gradient, orientation = getGradient(img)
            gradient = discreteNonMaximaSuppression(gradient, orientation)

        print(" Computing hough transform")

        rhoRes = RHO_RES
        thetaRes = THETA_RES
        hough = myHoughTransform(gradient, rhoRes, thetaRes)
        hough = rescaleIntensities(hough)
        writeImage(hough, os.path.join(out, filename+"_hough.jpg"))

        nLines = 10
        print(f" Finding top-n={nLines} lines")
        lines = myHoughLines(hough, nLines, rhoRes, thetaRes)

        print(" Drawing results")
        output = drawLines(gradient, lines)
        writeImage(output, os.path.join(out, filename+"_lines.jpg"))


def main():
    DIR = "./dataset"
    OUT = "./out"  # intermediate output images directory
    directory = os.fsencode(DIR)
    if not os.path.exists(OUT):
        os.makedirs(OUT)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        processImage(filename, data=DIR, out=OUT)


if __name__ == "__main__":
    main()

"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import matlab

import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 203201389


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    # https://stackoverflow.com/questions/51987844/what-is-the-representation-of-rgb-image

    img = cv2.imread(filename)  # read in BGR format
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # print("rows " + str(img.size))
    # print("cols " + str(img[0].size))

    if representation == 1:
        cv2.imwrite("res.jpg", img)
    elif representation == 2:
        temp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("res.jpg", temp)

    img = np.float32(img)
    img = (1 / 255) * img

    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    # https://stackoverflow.com/questions/52333972/opencv-convert-image-to-grayscale-and-display-using-matplotlib-gives-strange-co

    img = cv2.imread(filename)  # read in BGR format
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(img, cmap='gray')
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.show()
    pass


yiq_from_rgb = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    # (AB)^T = B^T * A^T

    # https://stackoverflow.com/questions/61348558/rgb-to-yiq-and-back-in-python
    # https://levelup.gitconnected.com/introduction-to-histogram-equalization-for-digital-image-enhancement-420696db9e43

    # original_dimensions = imgRGB.shape
    # yiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    # imgRGB = imgRGB.reshape(-1, 3)
    # # https://stackoverflow.com/questions/24560298/python-numpy-valueerror-operands-could-not-be-broadcast-together-with-shapes
    # # imgRGB = imgRGB * yiq
    # imgRGB = numpy.dot(imgRGB, yiq.transpose())
    # imgRGB = imgRGB.reshape(original_dimensions)
    # plt.imshow(imgRGB)
    # plt.show()
    # return imgRGB
    print(imgRGB.dtype)
    OrigShape = imgRGB.shape
    res = np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(OrigShape)
    # plt.imshow((res * 255).astype(np.uint8))
    # plt.show()
    return res


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # original_dimensions = imgYIQ.shape
    # yiq_inversed = np.linalg.inv(np.array([[0.299, 0.587, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]))
    # imgYIQ = imgYIQ.reshape(-1, 3)
    # # https://stackoverflow.com/questions/24560298/python-numpy-valueerror-operands-could-not-be-broadcast-together-with-shapes
    # # imgRGB = imgRGB * yiq
    # imgYIQ = numpy.dot(imgYIQ, yiq_inversed.transpose())
    # imgYIQ = imgYIQ.reshape(original_dimensions)
    OrigShape = imgYIQ.shape
    res = np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)
    # plt.imshow((res * 255).astype(np.uint8))
    # plt.show()
    return res


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """

    # https://towardsdatascience.com/histogram-equalization-a-simple-way-to-improve-the-contrast-of-your-image-bcd66596d815
    # https://stackoverflow.com/questions/61178379/how-to-do-histogram-equalization-without-using-cv2-equalizehist
    if len(imgOrig.shape) == 3:
        img_as_yiq = transformRGB2YIQ(imgOrig)
        img_as_yiq = 255.0 * img_as_yiq
        # 1. Compute the image histogram
        histOrg, bin = np.histogram(img_as_yiq[:, :, 0], 256, [0, 255])
        # 2. Compute the cumulative histogram
        cdf = np.cumsum(histOrg)
        # 3. Normalize the cumulative histogram (divide by the total number of pixels)
        cdf = cdf / np.sum(histOrg)
        # 4. Multiply the normalized cumulative histogram by the maximal gray level value (K-1)
        # 5. Round the values to get integers
        cdf = np.floor(255 * cdf).astype(np.uint8)
        # 6. Map the intensity values of the image using the result of step 5
        backup_shape = img_as_yiq[:, :, 0].shape
        flatten_img = list(img_as_yiq[:, :, 0].astype(int).flatten())
        equalized_flatten_img = [cdf[p] for p in flatten_img]
        img_as_yiq[:, :, 0] = np.reshape(np.asarray(equalized_flatten_img), backup_shape)
        histEq, bin = np.histogram(img_as_yiq[:, :, 0], 256, [0, 255])

        imgEq = transformYIQ2RGB(img_as_yiq * 1 / 255)
        return imgEq, histOrg, histEq
    else:
        img_as_yiq = 255.0 * imgOrig
        # 1. Compute the image histogram
        histOrg, bin = np.histogram(img_as_yiq, 256, [0, 255])
        # 2. Compute the cumulative histogram
        cdf = np.cumsum(histOrg)
        # 3. Normalize the cumulative histogram (divide by the total number of pixels)
        cdf = cdf / np.sum(histOrg)
        # 4. Multiply the normalized cumulative histogram by the maximal gray level value (K-1)
        # 5. Round the values to get integers
        cdf = np.floor(255 * cdf).astype(np.uint8)
        # 6. Map the intensity values of the image using the result of step 5
        backup_shape = img_as_yiq.shape
        flatten_img = list(img_as_yiq.astype(int).flatten())
        equalized_flatten_img = [cdf[p] for p in flatten_img]
        imgEq = np.reshape(np.asarray(equalized_flatten_img), backup_shape)
        histEq, bin = np.histogram(imgEq, 256, [0, 255])
        return imgEq, histOrg, histEq
    pass


def calc_pi(inx: int, z: np.ndarray, hist: np.ndarray) -> int:
    value = 0
    sum = 0
    for color in range(z[inx], z[inx + 1] + 1):
        value = value + (color * hist[color])
        sum = sum + hist[color]
    return value / sum


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if nQuant > 255 or nQuant < 1:
        return
    imOrig = imOrig * 255.0
    # Get histogram
    hist, bin = np.histogram(imOrig, 256, [0, 255])
    # Init z,p
    z = numpy.append(np.arange(0, 255, 255 / nQuant).astype(int), 255)  # example(2): [0,127,255]
    p = np.zeros(nQuant, dtype=int)  # example(2): [0,0]
    for m in range(0, nIter):
        p = [calc_pi(idx, z, hist) for idx, item in enumerate(p)]
        z_temp = [((p[idx] + p[idx + 1]) / 2) for idx, item in enumerate(z[1:-1].copy())]
        z = numpy.append(0, numpy.append(z_temp, 255)).astype(int)

    # set the new image
    backup_shape = imOrig.shape
    flatten_img = list(imOrig.astype(int).flatten())
    equalized_flatten_img = [p[int(x / (255 / nQuant)) - 1] for x in flatten_img]
    newImg = np.reshape(np.asarray(equalized_flatten_img), backup_shape)
    plt.imshow(newImg, cmap='gray')
    plt.title("newImg")
    plt.show()
    pass

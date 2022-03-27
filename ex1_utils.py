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
import math
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
    img = cv2.imread(filename)  # read in BGR format
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    img = cv2.imread(filename)  # read in BGR format
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(img, cmap='gray')
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # since matplotlib assumes RGB
        plt.imshow(img)
    plt.show()
    pass


yiq_from_rgb = np.array([[0.299, 0.587, 0.114], [0.595716, -0.274453, -0.321263], [0.211456, -0.522591, 0.311135]])


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    backup_shape = imgRGB.shape
    res = np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(backup_shape)
    # We need to multiply by 255.0 and then round to the nearest Integer number (with 'np.rint' func)
    # and then normalize again (divide by 255.0) in order to allow the image to contain a completely white color (255)
    return np.rint(res * 255.0) / 255.0


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    backup_shape = imgYIQ.shape
    res = np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(backup_shape)
    # We need to multiply by 255.0 and then round to the nearest Integer number (with 'np.rint' func)
    # and then normalize again (divide by 255.0) in order to allow the image to contain a completely white color (255)
    return np.rint(res * 255.0) / 255.0


"""
Histogram Equalization Algorithm:
    1. Compute the image histogram
    2. Compute the cumulative histogram
    3. Normalize the cumulative histogram (divide by the total number of pixels)
    4. Multiply the normalized cumulative histogram by the maximal gray level value (K-1)
    5. Round the values to get integers
    6. Map the intensity values of the image using the result of step 5
    7. Verify that the minimal value is 0 and that the maximal is K-1,
        otherwise stretch the result linearly in the range [0,K-1].

How to normalize the cumulative histogram and stretch it as much as possible (0 to 255):
    The idea is to multiply by 255 the relative position of each color in the original color range. 
    For example, if the image contains 1/3 pixels in color 137, 1/3 pixels in color 138 and 1/3 pixels in color 139. 
    Then we get a normalized pdf: pdf [137] = 1/3, pdf [138] = 2/3, pdf [ 139] = 1. 
    Hence, the original size interval is 3 and:
        The relative position (in the original interval) of 137 is 0. 
        The relative position (in the original interval) of 138 is 1/2 
        The relative position (in the original interval) of 139 is 1.
    That is, we will need to move color 138 to the relative position 1/2 in the final interval [0,255] which is 127. 
    (and same for the other colors).
    We will use the following formula:
    cdf = 255 * (cdf - np.min(cdf[np.nonzero(cdf)])) / (cdf.max() - np.min(cdf[np.nonzero(cdf)]))
    When we take the value of the minimum color (regardless of the values 0. 
    that is, the colors that are not in the image at all)
"""


def hsitogramEqualizeAlgo(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    histOrg, bin = np.histogram(img, 256, [0, 255])
    cdf = np.cumsum(histOrg)
    cdf = 255 * (cdf - np.min(cdf[np.nonzero(cdf)])) / (cdf.max() - np.min(cdf[np.nonzero(cdf)]))
    backup_shape = img.shape
    flatten_img = list(img.astype(int).flatten())
    equalized_flatten_img = [cdf[p] for p in flatten_img]
    img = np.reshape(np.asarray(equalized_flatten_img), backup_shape)
    histEq, bin = np.histogram(img, 256, [0, 255])
    return img, histOrg, histEq


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if len(imgOrig.shape) == 3:
        img_as_yiq = transformRGB2YIQ(imgOrig)
        img_as_yiq = 255.0 * img_as_yiq
        img_y = img_as_yiq[:, :, 0]
        img_y_Eq, histOrg, histEq = hsitogramEqualizeAlgo(img_y)
        img_as_yiq[:, :, 0] = img_y_Eq
        imgEq = transformYIQ2RGB(img_as_yiq * 1 / 255)
        return imgEq, histOrg, histEq
    else:
        img = 255.0 * imgOrig
        imgEq, histOrg, histEq = hsitogramEqualizeAlgo(img)
        imgEq = imgEq * 1 / 255
        return imgEq, histOrg, histEq
    pass


def calc_pi(inx: int, z: np.ndarray, hist: np.ndarray) -> int:
    value = sum = 0
    for color in range(z[inx], z[inx + 1] + 1):
        value = value + (color * hist[color])
        sum = sum + hist[color]
    if sum == 0:
        return 0
    return np.rint(value / sum)


def quantizeImageAlgo(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    # Get histogram
    hist, bin = np.histogram(imOrig, 256, [0, 255])
    z = numpy.append(np.arange(0, 255, 255 / nQuant).astype(int), 255)  # example(2): [0,127,255]
    p = np.zeros(nQuant, dtype=int)  # example(2): [0,0]
    image_list = []
    error_list = []
    for m in range(0, nIter):
        p = [calc_pi(idx, z, hist) for idx, item in enumerate(p)]
        z_temp = [((p[idx] + p[idx + 1]) / 2) for idx, item in enumerate(z[1:-1])]  # .copy()
        z = numpy.append(0, numpy.append(z_temp, 255)).astype(int)
        error = 0
        for i in range(0, nQuant):
            for color in range(z[i], z[i + 1] + 1):
                error = error + (np.subtract(p[i], color) ** 2 * hist[color])
        error_list.append(error)
        backup_shape = imOrig.shape
        flatten_img = list(imOrig.astype(int).flatten())
        equalized_flatten_img = [p[int(x / (256 / nQuant))] for x in flatten_img]
        newImg = np.reshape(np.asarray(equalized_flatten_img), backup_shape)
        image_list.append(newImg)

    return image_list, error_list


def yiqListToRGB(img_as_yiq: np.ndarray, img_y: np.ndarray) -> np.ndarray:
    img_as_yiq[:, :, 0] = img_y
    return transformYIQ2RGB(img_as_yiq * 1 / 255)


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

    if len(imOrig.shape) == 3:
        img_as_yiq = transformRGB2YIQ(imOrig)
        img_as_yiq = 255.0 * img_as_yiq
        img_y = img_as_yiq[:, :, 0]
        image_list, error_list = quantizeImageAlgo(img_y, nQuant, nIter)
        image_list = [yiqListToRGB(img_as_yiq, image) for image in image_list]
        return image_list, error_list
    else:
        img = 255.0 * imOrig
        image_list, error_list = quantizeImageAlgo(img, nQuant, nIter)
        image_list = [(image * 1 / 255) for image in image_list]
        return image_list, error_list
    pass

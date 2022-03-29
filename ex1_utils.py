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
import numpy
from bisect import bisect_left

import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

'''
#################################################################################################################
################################################# myID METHOD ###################################################
#################################################################################################################
'''


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 203201389


'''
#################################################################################################################
########################################### imReadAndConvert METHOD #############################################
#################################################################################################################
'''


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation != 1 and representation != 2:
        print("Invalid representation value (Mandatory conditions: 1<=representation<=2).")
        exit(1)
    img = np.array
    try:
        img = cv2.imread(filename)  # read in BGR format
        if representation == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif representation == 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # since matplotlib assumes RGB
        img = np.float32(img)
        img = (1 / 255) * img
    except (Exception,):
        print("An exception occurred: can't open/read file: check file path/integrity")
        exit(1)

    return img


'''
#################################################################################################################
############################################### imDisplay METHOD ################################################
#################################################################################################################
'''


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    if representation != 1 and representation != 2:
        print("Invalid representation value (Mandatory conditions: 1<=representation<=2).")
        exit(1)
    try:
        img = imReadAndConvert(filename, representation)
        if representation == 1:
            plt.imshow(img, cmap='gray')
            plt.show()
        elif representation == 2:
            plt.imshow(img)
            plt.show()
    except (Exception,):
        print("An exception occurred: can't open/read file: check file path/integrity")
    pass


'''
#################################################################################################################
########################################### transformRGB2YIQ METHOD #############################################
#################################################################################################################
'''

'''

Given the red (R), green (G), and blue (B) pixel components of an RGB color image,
the corresponding luminance (Y), and the chromaticity components (I and Q) in the YIQ color space are
linearly related as follows:

  _ _       __                   __     _ _
 | Y |     | 0.299,  0.587,  0.114 |   | R |
 |   |     |                       |   |   |
 | I |  =  | 0.596, -0.275, -0.321 | X | G |
 |   |     |                       |   |   |
 | Q |     | 0.212, -0.523,  0.311 |   | B |
 |_ _|     |__                   __|   |_ _|
 
 Explanation:
    We will first flatten the matrix so that we can get an order matrix (X, 3). 
    And so we can make a multiplication to the right of each pixel in the transpose matrix 
    of the transition matrix [which is an order (3, 3)].
    In this way we will get the same result obtained from performing the formula above for each individual pixel.
'''


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    try:
        backup_shape = imgRGB.shape
        res = np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(backup_shape)
        # We need to multiply by 255.0 and then round to the nearest Integer number (with 'np.rint' func) and then
        # normalize again (divide by 255.0) in order to allow the image to contain a completely white color (255)
        return np.rint(res * 255.0) / 255.0
    except (Exception,):
        print("An exception occurred: can't convert the image from RGB to YIQ")
        exit(1)


'''
#################################################################################################################
########################################### transformYIQ2RGB METHOD #############################################
#################################################################################################################
'''
'''

Given the luminance (Y) and the chromaticity (I and Q) pixel components of an YIQ color image,
the corresponding red (R), green (G), and blue (B) pixel components in the RGB color space are
linearly related as follows:

  _ _       __                   __ -1    _ _
 | R |     | 0.299,  0.587,  0.114 |     | Y |
 |   |     |                       |     |   |
 | G |  =  | 0.596, -0.275, -0.321 |  X  | I |
 |   |     |                       |     |   |
 | B |     | 0.212, -0.523,  0.311 |     | Q |
 |_ _|     |__                   __|     |_ _|
 
! Note the marking (-1) that expresses an inverted matrix !

Explanation: Same as written in the 'transformRGB2YIQ' method above.
'''


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    try:
        backup_shape = imgYIQ.shape
        res = np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(backup_shape)
        # We need to multiply by 255.0 and then round to the nearest Integer number (with 'np.rint' func) and then
        # normalize again (divide by 255.0) in order to allow the image to contain a completely white color (255)
        return np.rint(res * 255.0) / 255.0
    except (Exception,):
        print("An exception occurred: can't convert the image from RGB to YIQ")
        exit(1)


'''
#################################################################################################################
########################################### hsitogramEqualize METHOD ############################################
#################################################################################################################
'''

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


def __hsitogramEqualizeAlgo(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    img = (255.0 * imOrig).astype('uint8')
    histOrg, bin = np.histogram(img, 256, [0, 255])
    cdf = np.cumsum(histOrg)
    cdf = 255 * (cdf - np.min(cdf[np.nonzero(cdf)])) / (cdf.max() - np.min(cdf[np.nonzero(cdf)]))
    backup_shape = img.shape
    flatten_img = list(img.astype(int).flatten())
    equalized_flatten_img = [cdf[p] for p in flatten_img]
    img = np.reshape(np.asarray(equalized_flatten_img), backup_shape)
    histEq, bin = np.histogram(img, 256, [0, 255])
    return img * 1 / 255, histOrg, histEq


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    try:
        img_shape = imgOrig.shape
    except (Exception,):
        print("An exception occurred: invalid image array.")
        exit(1)
    if len(img_shape) == 3:
        img_as_yiq = transformRGB2YIQ(imgOrig)
        img_y = img_as_yiq[:, :, 0]
        img_y_Eq, histOrg, histEq = __hsitogramEqualizeAlgo(img_y)
        img_as_yiq[:, :, 0] = img_y_Eq
        imgEq = transformYIQ2RGB(img_as_yiq)
        return imgEq, histOrg, histEq
    elif len(img_shape) == 2:
        imgEq, histOrg, histEq = __hsitogramEqualizeAlgo(imgOrig)
        return imgEq, histOrg, histEq
    pass


'''
#################################################################################################################
############################################# quantizeImage METHOD ##############################################
#################################################################################################################
'''

"""
Building look up table after each iteration:
'bisect_left' method finds the first position at which an element could be inserted 
in a given sorted range while maintaining the sorted order.
After each relocation to the 'p' & 'z' arrays:
    For each color were using 'bisect_left' method to search the index of the "Container" that represent the new color
    in the 'z' array (the array that divides the color boundaries).
    For example, if z = [0 86 167 255] then there are 3 Container (because 4 boundaries represent 3 Containers)
    and the color 87 belongs to the second container.
    Hence, bisect_left(z, 87, 1, None) = 2.
    But the count of the indexes starts from 0 and therefore we subtract 1 from the result.
    [We start checking from index 1 to the end (none). 
    The reason we check from index 1 is to avoid getting a negative value in the case of a color with a value of 0]
"""


def __quantizeImageAlgo(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    img = (255.0 * imOrig).astype('uint8')
    hist, bin = np.histogram(img, 256, [0, 255])
    z = numpy.append(np.arange(0, 255, 255 / nQuant).astype(int), 255)  # example(2): [0,127,255]
    p = np.zeros(nQuant, dtype=int)  # example(2): [0,0]
    image_list = []
    error_list = []
    colors_range = np.array(np.arange(0, 256))
    for _ in range(nIter):
        # calculate z,p
        p = [__calc_pi(idx, z, hist) for idx, item in enumerate(p)]
        z_temp = [((p[idx] + p[idx + 1]) / 2) for idx, item in enumerate(z[1:-1])]  # .copy()
        z = numpy.append(0, numpy.append(z_temp, 255)).astype(int)
        # calculate the error
        error = 0
        for i in range(nQuant):
            for color in range(z[i], z[i + 1] + 1):
                error += ((p[i] - color) ** 2 * hist[color])
        error_list.append(error)
        # calculate the new image
        look_up_table = np.array([p[bisect_left(z, x, 1, None) - 1] for x in colors_range])
        newImg = cv2.LUT(img, look_up_table.astype("uint8"))
        image_list.append(newImg * 1 / 255)
    return image_list, error_list


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if nQuant > 255 or nQuant < 1 or nIter < 0:
        print("Invalid Arguments to quantizeImage method (Mandatory conditions: 1<=nQuant<=255 and nIter>=0).")
        exit(1)
    if nIter == 0:
        return [imOrig], [0]
    try:
        img_shape = imOrig.shape
    except (Exception,):
        print("An exception occurred: invalid image array.")
        exit(1)
    # Execute the algorithm according to the type of image
    if len(img_shape) == 3:
        img_as_yiq = transformRGB2YIQ(imOrig)
        image_list, error_list = __quantizeImageAlgo(img_as_yiq[:, :, 0], nQuant, nIter)
        image_list = [__yiqListToRGB(img_as_yiq, image) for image in image_list]
        return image_list, error_list
    elif len(img_shape) == 2:
        image_list, error_list = __quantizeImageAlgo(imOrig, nQuant, nIter)
        return image_list, error_list
    pass


'''
__calc_pi method:
Auxiliary method to '__ quantizeImageAlgo'.
Calculation of the color value in each "container".
'''


def __calc_pi(inx: int, z: np.ndarray, hist: np.ndarray) -> int:
    value = sum = 0
    for color in range(z[inx], z[inx + 1] + 1):
        value = value + (color * hist[color])
        sum = sum + hist[color]
    if sum == 0:
        return 0
    return np.rint(value / sum)


'''
__yiqListToRGB method:
Auxiliary method to ''quantizeImage'.
Rebuild the images by re-integrating the new Y channel into the original image.
'''


def __yiqListToRGB(img_as_yiq: np.ndarray, img_y: np.ndarray) -> np.ndarray:
    img_as_yiq[:, :, 0] = img_y
    return transformYIQ2RGB(img_as_yiq)


'''
#################################################################################################################
################################################# That's it! ####################################################
#################################################################################################################

░░░░░░░░░░░░░░░░░░░░░░██████████████░░░░░░░░░
░░███████░░░░░░░░░░███▒▒▒▒▒▒▒▒▒▒▒▒▒███░░░░░░░
░░█▒▒▒▒▒▒█░░░░░░░███▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒███░░░░
░░░█▒▒▒▒▒▒█░░░░██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██░░
░░░░█▒▒▒▒▒█░░░██▒▒▒▒▒██▒▒▒▒▒▒▒▒▒▒▒██▒▒▒▒▒███░
░░░░░█▒▒▒█░░░█▒▒▒▒▒▒████▒▒▒▒▒▒▒▒▒████▒▒▒▒▒▒██
░░░█████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██
░░░█▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒▒▒███▒▒▒▒▒▒▒▒▒▒▒▒██
░██▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██▒▒▒▒██
██▒▒▒███████████▒▒▒▒▒██▒▒▒▒▒▒▒▒▒▒▒▒▒██▒▒▒▒▒██
█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒█████████████▒▒▒▒▒▒▒██
██▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██░
░█▒▒▒███████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██░░░
░██▒▒▒▒▒▒▒▒▒▒████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█░░░░░
░░████████████░░░██████████████████████░░░░░░
'''

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
import numpy as np

from ex1_utils import LOAD_GRAY_SCALE, imReadAndConvert
import cv2

global img

'''
'on_trackbar' method:
    Auxiliary method to 'gammaDisplay'.
    Performs a recalculation of the gamma function whenever the value of the slider changes.
'''


def on_trackbar(val):
    gamma = val / 100
    colors = np.arange(0, 256)
    table = np.array([255 * ((i / 255.0) ** gamma) for i in colors]).astype("uint8")
    # apply gamma correction using the lookup table
    new_img = cv2.LUT(img, table)
    title = 'Gamma Correction (current gamma value = %.2f)' % float("{:.2f}".format(val / 100.0))
    cv2.setWindowTitle('Gamma Correction', title)
    cv2.imshow('Gamma Correction', new_img)


'''
'gammaDisplay' method:
    performs gamma correction on an image with a given 'gamma'.
'''


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    if rep != 1 and rep != 2:
        return
    global img
    try:
        img = imReadAndConvert(img_path, rep)
        if rep == 1:
            img = cv2.imread(img_path, 0)
        else:
            img = cv2.imread(img_path)
    except (Exception,):
        print("An exception occurred")
        return

    cv2.namedWindow('Gamma Correction')
    trackbar_name = 'Gamma'
    cv2.createTrackbar(trackbar_name, 'Gamma Correction', 0, 200, on_trackbar)
    cv2.setTrackbarPos(trackbar_name, 'Gamma Correction', 100)
    # cv2.moveWindow('Gamma Correction', 50, 50)
    on_trackbar(100)
    # Wait until user press some key
    cv2.waitKey()

    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()

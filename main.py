# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
from matplotlib.pyplot import subplot

from ex1_utils import *
from gamma import gammaDisplay


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def quantizeImageTest(imOrig: np.ndarray, nQuant: int, nIter: int):
    image_list, error_list = quantizeImage(imOrig, nQuant, nIter)
    print(str("image_list size: ") + str(len(image_list)))
    print(str("error_list size: ") + str(len(error_list)))

    plt.plot(error_list)
    plt.title("error_list")
    plt.show()

    # show result
    if len(image_list[len(image_list) - 1].shape) == 3:
        plt.imshow(image_list[len(image_list) - 1])
    else:
        plt.imshow(image_list[len(image_list) - 1], cmap='gray')
    plt.title("newImg")
    plt.show()


def hsitogramEqualizeTest(imOrig: np.ndarray):
    imgEq, histOrg, histEq = hsitogramEqualize(imOrig)
    plt.xlim([0, 256])
    plt.plot(histOrg)
    plt.title("histOrg")
    plt.show()
    plt.plot(histEq)
    plt.title("histEq")
    plt.show()
    plt.imshow(imgEq, cmap='gray')
    # plt.imshow(imgEq)
    plt.title("image")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print("Your OpenCV version is: " + cv2.__version__)
    # # img_path = 'beach.jpg'
    # # img_path = 'pout.tif'
    img_path = 'Lenna.png'
    # img_path = 'gray.jpg'

    # # Basic read and display
    img = imReadAndConvert(img_path, 2)
    # imDisplay(img_path, LOAD_GRAY_SCALE)
    # im = imReadAndConvert(img_path, 2)
    # imDisplay(img_path, LOAD_RGB)

    # # im = transformRGB2YIQ(im)
    # # im = transformYIQ2RGB(im)

    # im = imReadAndConvert(img_path, 2)
    # hsitogramEqualizeTest(im)

    im = imReadAndConvert('dog.jpg', 2)
    quantizeImageTest(im, 3, 10)

    # gammaDisplay('fall.jpg', 2)
    # gammaDisplay('Lenna.png', 1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

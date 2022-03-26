# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
from matplotlib.pyplot import subplot

from ex1_utils import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    # print("Your OpenCV version is: " + cv2.__version__)
    # # img_path = 'beach.jpg'
    # # img_path = 'pout.tif'
    # img_path = 'Lenna.png'
    img_path = 'gray.jpg'
    #
    # # Basic read and display
    # im = imReadAndConvert(img_path, 1)
    # imDisplay(img_path, LOAD_GRAY_SCALE)
    # im = imReadAndConvert(img_path, 2)
    # imDisplay(img_path, LOAD_RGB)
    #
    im = imReadAndConvert(img_path, 1)
    # # im = transformRGB2YIQ(im)
    # # im = transformYIQ2RGB(im)
    # imgEq, histOrg, histEq = hsitogramEqualize(im)
    #
    # # plt.plot()
    # # width = 0.7 * (bins[1] - bins[0])
    # # center = (bins[:-1] + bins[1:]) / 2
    # plt.xlim([0, 256])
    # plt.plot(histOrg)
    # plt.title("histOrg")
    # plt.show()
    # plt.plot(histEq)
    # plt.title("histEq")
    # plt.show()
    # plt.imshow(imgEq, cmap='gray')
    # # plt.imshow(imgEq)
    # plt.title("image")
    # plt.show()

    im = imReadAndConvert('dog.jpg', 1)
    quantizeImage(im, 6, 100)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

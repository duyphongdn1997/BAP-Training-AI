import cv2
import numpy as np
import glob
import os
from PIL import Image


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))


def text_detection(image_path: str = "./image/"):
    """

    :param image_path:
    :return:
    """
    os.chdir(image_path)
    images = []
    grays = []
    for file in glob.glob("*.png"):
        img = cv2.imread(file)
        images.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grays.append(gray)

    edges = cv2.Canny(images[0], 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if 10000 > cv2.contourArea(cnt) > 30:
            x, y, w, h = cv2.boundingRect(cnt)
            x2, y2 = x + w, y + h
            cv2.rectangle(images[0], (x, y), (x2, y2), (0, 0, 255), 3)
    # cv2.drawContours(images[0], contours, -1, (0, 0, 255), 3)
    Image.fromarray(images[0]).show()
    # cv2.imshow("image", images[0])
    # cv2.resizeWindow("image", 960, 540)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    text_detection()

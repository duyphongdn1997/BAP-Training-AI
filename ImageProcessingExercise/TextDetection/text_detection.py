import cv2
import numpy as np
import glob
from random import randint


class TextDetector(object):
    """
    This class used to perform examination image processing.
    """
    image = None

    def __init__(self, kernel_size: int = 3):
        """
        This is the init
        :param kernel_size: The size of kernel use to process.
        """
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        self.contour = []
        self.images = []
        self.set_image()

    def set_image(self, image_path: str = "./image/",
                  image_name: str = "0.png"):
        """
        This method used to read image input to BGR format.
        :param image_path: Path to the image
        :param image_name: Name of the image that want to read in "path"
        """
        self.image = cv2.imread(image_path + image_name)
        print("Read image done!")

    def get_image(self) -> np.ndarray:
        """
        This method return the image (BGR format)
        :return: Image in BGR format.
        """
        return self.image

    def set_many_image(self, image_path: str = "./image/"):
        """
        This method used to read many image and store them in a list.
        :param image_path:
        """
        for file in glob.glob(image_path + "*.png"):
            image = cv2.imread(file)
            self.images.append(image)

    @staticmethod
    def canny_edge_detection(image, blur_k_size=5, threshold1=100, threshold2=300, cvt_color=True):
        if cvt_color:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_gaussian = cv2.GaussianBlur(gray, (blur_k_size, blur_k_size), 1)
        else:
            img_gaussian = cv2.GaussianBlur(image, (blur_k_size, blur_k_size), 1)
        img_canny = cv2.Canny(img_gaussian, threshold1, threshold2)
        return img_canny

    def __get_dilate(self, iterations: int = 5):
        """
        This method used to getting dilation of the image.
        :return: dilation of image
        """
        img = self.get_image()
        edges = self.canny_edge_detection(img)
        dilate = cv2.dilate(edges, self.kernel, iterations=iterations)
        return dilate

    def text_detection(self, crop: bool = True):
        """
        This method used to detect text in only one image
        """
        dilate = self.__get_dilate()
        img = self.get_image()
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_ = filter(lambda contour: cv2.boundingRect(contour)[1] > 150 and cv2.boundingRect(contour)[3] < 100
                           and cv2.contourArea(contour) > 100, contours)
        mask = np.zeros_like(dilate)
        for cnt in contours_:
            x, y, w, h = cv2.boundingRect(cnt)
            x2, y2 = x + w, y + h
            pts = np.array([[x, y], [x, y2], [x2, y2], [x2, y]])
            cv2.fillPoly(mask, pts=[pts], color=(255, 255, 255))

        mask = cv2.dilate(mask, self.kernel, iterations=2)
        if crop:
            crop_image = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow("Crop Image", crop_image)
        else:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def big_question_detector(self, draw: bool = True, crop: bool = True):
        """
        This method used to detect big question contours
        :param crop: Set True order to show crop only big question, False if show whole with big question contours.
        :param draw: Draw or not draw
        :return: List of contours of the big questions
        """
        dilate = self.__get_dilate()
        img = self.get_image()
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_ = list(filter(lambda cnt: cv2.boundingRect(cnt)[1] > 150 and cv2.boundingRect(cnt)[3] > 100
                                and cv2.contourArea(cnt) > 1600 and cv2.boundingRect(cnt)[2] > 1000, contours))
        if draw:
            if crop:
                mask = np.zeros_like(dilate)
                cv2.fillPoly(mask, pts=contours_, color=(255, 255, 255))
                crop_image = cv2.bitwise_and(img, img, mask=mask)
                cv2.imshow("Crop Image", crop_image)
            else:
                cv2.drawContours(img, contours_, -1, (0, 0, 255), 3)
                cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            return contours_

    @staticmethod
    def add_tictoe(positions: list, image: np.ndarray) -> np.ndarray:
        """
        This method used to add tic toe to our image with defined positions
        :param positions: position of the tictoe.
        :param image: The input image
        :return: The image has some tictoe
        """
        for pos in positions:
            cv2.drawMarker(image, pos, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=50,
                           thickness=3, line_type=cv2.LINE_AA)
        return image

    def none_tictoe_ignore(self):
        """
        This method used to draw contours of big questions that has tic toe inside it.
        """
        img = self.get_image()
        contours = self.big_question_detector(draw=False)
        positions = [(randint(150, 1700), randint(300, 2700)) for i in range(20)]
        img_tictoe = self.add_tictoe(positions=positions, image=img)
        contours_ = []
        for contour in contours:
            distances = [cv2.pointPolygonTest(contour, pos, False) for pos in positions]
            for dist in distances:
                if dist > 0:
                    contours_.append(contour)
                    break
        cv2.drawContours(img_tictoe, contours_, -1, (0, 0, 255), 3)
        cv2.imshow("Image", img_tictoe)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    text_detector = TextDetector()
    # text_detector.text_detection()
    text_detector.big_question_detector()
    # text_detector.none_tictoe_ignore()

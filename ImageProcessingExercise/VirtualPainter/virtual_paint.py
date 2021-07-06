import cv2 
import numpy as np
import pyautogui
from math import sqrt


class VirtualPaint:
    """
    This is Virtual Paint class
    """
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(self.camera_id)

    @staticmethod
    def get_color_segmentation(image: np.ndarray, lower_thresh: np.ndarray,
                               upper_thresh: np.ndarray):
        """
        This method used to get color segmentation.
        :param image: image input
        :param lower_thresh: lower thresh of color need to segment.
        :param upper_thresh: upper thresh of color need to segment.
        :return: mask of segmentation
        """
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_thresh, upper_thresh)
        return mask

    @staticmethod
    def mask_preprocess(mask: np.ndarray, erosion_kernel: np.ndarray,
                        dilation_kernel: np.ndarray,
                        erosion_iterations: int = 1,
                        dilation_iterations: int = 1):
        """
        This method used to preprocess the mask to reduce noise.
        :param mask: The mask input
        :param erosion_kernel: kernel to erode
        :param dilation_kernel: kernel to dilate
        :param erosion_iterations: iterations of erode
        :param dilation_iterations: iteration of dilate
        :return:
        """
        mask = cv2.erode(mask, erosion_kernel, erosion_iterations)
        mask = cv2.dilate(mask, dilation_kernel, dilation_iterations)
        return mask

    def virtual_painter(self):
        """
        This method used to run virtual paint.
        """
        line = None
        x_before_red, y_before_red, x_after_red, y_after_red = 0, 0, 0, 0
        x_before_yellow, y_before_yellow, x_after_yellow, y_after_yellow = 0, 0, 0, 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        while True:
            ret, frame = self.camera.read()
            frame = cv2.flip(frame, flipCode=1)
            if line is None:
                line = np.zeros_like(frame)

            # Threshold the HSV image to get only blue and yellow colors
            lower_red = np.array([176, 217, 0])
            upper_red = np.array([179, 255, 254])

            lower_yellow = np.array([28, 101, 30])
            upper_yellow = np.array([33, 255, 255])

            yellow_mask = self.get_color_segmentation(frame, lower_yellow, upper_yellow)
            red_mask = self.get_color_segmentation(frame, lower_red, upper_red)
            yellow_mask = self.mask_preprocess(yellow_mask, erosion_kernel=kernel,
                                               dilation_kernel=kernel,
                                               erosion_iterations=1,
                                               dilation_iterations=5)

            red_mask = self.mask_preprocess(red_mask, erosion_kernel=kernel,
                                            dilation_kernel=kernel_,
                                            erosion_iterations=1,
                                            dilation_iterations=150)

            # Find contours
            red_contours, hierarchy_red = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            yellow_contours, hierarchy_yellow = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Threshold noise
            # cv2.contourArea to take area of contour
            # key parameter used to apply cv2.contourArea to contours before using max()

            try:
                non_noise_red = cv2.contourArea(max(red_contours, key=cv2.contourArea))
                # non_noise_yellow = cv2.contourArea(max(yellow_contours, key=cv2.contourArea))
            except ValueError:
                print("None of red contours")
                
            try:
                # non_noise_red = cv2.contourArea(max(red_contours, key=cv2.contourArea))
                non_noise_yellow = cv2.contourArea(max(yellow_contours, key=cv2.contourArea))
            except ValueError:
                print("None of yellow contours") 

            # Detect red and yellow objects
            if red_contours and non_noise_red > 450:
                red_contour = max(red_contours, key=cv2.contourArea)
                x_after_red, y_after_red, w, h = cv2.boundingRect(red_contour)
                if x_before_red == 0 and y_before_red == 0:
                    x_before_red, y_before_red = x_after_red, y_after_red
                else:
                    line = cv2.line(line, (x_before_red, y_before_red),
                                    (x_after_red, y_after_red), [0, 0, 255], thickness=7)

            if yellow_contours and non_noise_yellow > 500:
                yellow_contour = max(yellow_contours, key=cv2.contourArea)
                x_after_yellow, y_after_yellow, w, h = cv2.boundingRect(yellow_contour)
                if x_before_yellow == 0 and y_before_yellow == 0:
                    x_before_yellow, y_before_yellow = x_after_yellow, y_after_yellow
                else:
                    line = cv2.line(line, (x_before_yellow, y_before_yellow),
                                    (x_after_yellow, y_after_yellow), [0, 255, 255], thickness=7)

            x_before_yellow, y_before_yellow = x_after_yellow, y_after_yellow
            x_before_red, y_before_red = x_after_red, y_after_red

            # Visualize mask
            red_yellow_mask = cv2.bitwise_and(frame, frame, mask=red_mask + yellow_mask)

            frame = cv2.add(frame, line)
            stacked = np.hstack((frame, red_yellow_mask))
            cv2.imshow("Virtual Painter", stacked)

            if cv2.waitKey(1) & 0xFF == ord("x"):
                line = np.zeros_like(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.camera.release()
        cv2.destroyAllWindows()

    def volume_control(self, scale: float = 0.0001):
        """
        This method used to control the volume of the PC using distance of two color of points.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        x_before_red, y_before_red, x_after_red, y_after_red = 0, 0, 0, 0
        x_before_yellow, y_before_yellow, x_after_yellow, y_after_yellow = 0, 0, 0, 0
        kernel_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        while True:
            ret, frame = self.camera.read()
            frame = cv2.flip(frame, flipCode=1)

            # Threshold the HSV image to get only blue and yellow colors
            lower_red = np.array([176, 100, 0])
            upper_red = np.array([179, 255, 254])

            lower_yellow = np.array([28, 101, 30])
            upper_yellow = np.array([33, 255, 255])

            yellow_mask = self.get_color_segmentation(frame, lower_yellow, upper_yellow)
            red_mask = self.get_color_segmentation(frame, lower_red, upper_red)
            yellow_mask = self.mask_preprocess(yellow_mask, erosion_kernel=kernel,
                                               dilation_kernel=kernel,
                                               erosion_iterations=1,
                                               dilation_iterations=5)

            red_mask = self.mask_preprocess(red_mask, erosion_kernel=kernel,
                                            dilation_kernel=kernel,
                                            erosion_iterations=1,
                                            dilation_iterations=200)

            # Find contours
            red_contours, hierarchy_red = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            yellow_contours, hierarchy_yellow = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Threshold noise
            # cv2.contourArea to take area of contour
            # key parameter used to apply cv2.contourArea to contours before using max()
            try:
                non_noise_red = cv2.contourArea(max(red_contours, key=cv2.contourArea))
                # non_noise_yellow = cv2.contourArea(max(yellow_contours, key=cv2.contourArea))
            except ValueError:
                print("None of red contours")
                
            try:
                # non_noise_red = cv2.contourArea(max(red_contours, key=cv2.contourArea))
                non_noise_yellow = cv2.contourArea(max(yellow_contours, key=cv2.contourArea))
            except ValueError:
                print("None of yellow contours")    
            distance_before = sqrt((x_before_red - x_before_yellow) ** 2 +
                                   (y_before_yellow - y_before_red) ** 2)

            # Detect red and yellow objects
            if red_contours and non_noise_red > 500:
                red_contour = max(red_contours, key=cv2.contourArea)
                x_after_red, y_after_red, w, h = cv2.boundingRect(red_contour)
                if x_before_red == 0 and y_before_red == 0:
                    x_before_red, y_before_red = x_after_red, y_after_red
                else:
                    cv2.rectangle(frame, (x_after_red, y_after_red),
                                  (x_after_red + w, y_after_red + h), [0, 0, 255], 3)

            if yellow_contours and non_noise_yellow > 500:
                yellow_contour = max(yellow_contours, key=cv2.contourArea)
                x_after_yellow, y_after_yellow, w, h = cv2.boundingRect(yellow_contour)
                if x_before_yellow == 0 and y_before_yellow == 0:
                    x_before_yellow, y_before_yellow = x_after_yellow, y_after_yellow
                else:
                    cv2.rectangle(frame, (x_after_yellow, y_after_yellow),
                                  (x_after_yellow + w, y_after_yellow + h), [0, 255, 255], 3)

            distance_after = sqrt((x_after_red - x_after_yellow) ** 2 +
                                  (y_after_yellow - y_after_red) ** 2)

            x_before_yellow, y_before_yellow = x_after_yellow, y_after_yellow
            x_before_red, y_before_red = x_after_red, y_after_red

            diff = distance_after - distance_before
            if diff >= 0:
                volume_up = int(diff * scale)
                for i in range(volume_up):
                    try:
                        pyautogui.press("volumeup")
                    except:
                        print("Nothing change!")
            else:
                volume_down = int(diff * scale)
                for i in range(volume_down):
                    try:
                        pyautogui.press("volumedown")
                    except:
                        print("Nothing change!")

            # Visualize mask
            red_yellow_mask = cv2.bitwise_and(frame, frame, mask=red_mask + yellow_mask)

            stacked = np.hstack((frame, red_yellow_mask))
            cv2.putText(stacked, "Volume :" + str(diff), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=[255, 0, 0], thickness=3)
            cv2.imshow("Control Volume", stacked)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    v = VirtualPaint(camera_id=0)
    v.volume_control()

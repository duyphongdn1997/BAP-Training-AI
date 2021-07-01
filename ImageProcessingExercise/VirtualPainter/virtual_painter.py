import cv2
import numpy as np
import argparse


def virtual_painter(opt):
    """
    Virtual painter
    :param opt: Option param
    """
    camera = cv2.VideoCapture(opt.camera_id)

    # Kernel for reduce noise
    noise = 0
    kernel = np.ones((5, 5), np.uint8)
    line = None
    x_before_red, y_before_red, x_after_red, y_after_red = 0, 0, 0, 0
    x_before_yellow, y_before_yellow, x_after_yellow, y_after_yellow = 0, 0, 0, 0

    while True:
        noise = 0
        ret, frame = camera.read()
        frame = cv2.flip(frame, flipCode=1)
        if line is None:
            line = np.zeros_like(frame)
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        # Notes: The range of hue 0-180° in OpenCV

        # Define range of blue color in HSV
        lower_red = np.array([155, 73, 43])
        upper_red = np.array([179, 255, 244])

        # Threshold the HSV image to get only blue and yellow colors
        lower_yellow = np.array([30, 110, 0])
        upper_yellow = np.array([40, 255, 255])

        mask_red = cv2.inRange(hsv_img, lower_red, upper_red)
        mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        mask = mask_yellow + mask_red

        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask_red = cv2.erode(mask_red, kernel, iterations=3)
        mask_red = cv2.dilate(mask_red, kernel, iterations=2)
        mask_yellow = cv2.erode(mask_yellow, kernel, iterations=1)
        mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=2)

        # Find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours_red, hierarchy_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_yellow, hierarchy_yellow = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Threshold noise
        # cv2.contourArea to take area of contour
        # key parameter used to apply cv2.contourArea to contours before using max()

        try:
            noise = cv2.contourArea(max(contours, key=cv2.contourArea))
        except ValueError as e:
            print("None of contours")

        # Detect blue and yellow object
        if contours_red and noise > 500:
            cnt_ = max(contours_red, key=cv2.contourArea)
            x_after_red, y_after_red, w, h = cv2.boundingRect(cnt_)
            print()
            if x_before_red == 0 and y_before_red == 0:
                x_before_red, y_before_red = x_after_red, y_after_red
            else:
                line = cv2.line(line, (x_before_red, y_before_red),
                                (x_after_red, y_after_red), [0, 0, 255], 7)

        if contours_yellow and noise > 500:
            cnt = max(contours_yellow, key=cv2.contourArea)
            x_after_yellow, y_after_yellow, w, h = cv2.boundingRect(cnt)
            if x_before_yellow == 0 and y_before_yellow == 0:
                x_before_yellow, y_before_yellow = x_after_yellow, y_after_yellow
            else:
                line = cv2.line(line, (x_before_yellow, y_before_yellow),
                                (x_after_yellow, y_after_yellow), [0, 255, 255], 7)

        x_before_yellow, y_before_yellow = x_after_yellow, y_after_yellow
        x_before_red, y_before_red = x_after_red, y_after_red

        # Visualize the mask
        blue_yellow = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert mask to 3-channel image
        mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame = cv2.add(frame, line)
        stacked = np.hstack((frame, blue_yellow))

        cv2.imshow("abc", stacked)

        if cv2.waitKey(1) & 0xFF == ord("x"):
            line = np.zeros_like(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--camera_id", default=0, type=int, help="Choose camera id")
    opt = parser.parse_args()
    virtual_painter(opt)

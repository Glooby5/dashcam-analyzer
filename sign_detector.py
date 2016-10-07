import numpy as np
import cv2


class SignDetector:

    @staticmethod
    def detectRedInRgb(image):
        lower = np.array([15, 15, 100], dtype=np.uint8)
        upper = np.array([90, 70, 255], dtype=np.uint8)

        mask = cv2.inRange(image, lower, upper)

        return mask

    @staticmethod
    def detectRedInHsv(image):
        lower = np.array([150, 70, 70], dtype=np.uint8)
        upper = np.array([210, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(image, lower, upper)

        return mask

    @staticmethod
    def getSignCircles(image):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=20, minRadius=22, maxRadius=50)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            return circles

        return None

    @staticmethod
    def highlightCircles(circles, image):
        if circles is None:
            return image

        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        return image

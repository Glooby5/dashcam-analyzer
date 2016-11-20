import numpy as np
import cv2


class SignDetector:
    RECTANGLE_WIDTH = 10
    COLOR_THRESHOLD = 1400

    @staticmethod
    def detectRedInRgb(image):
        lower = np.array([15, 15, 100], dtype=np.uint8)
        upper = np.array([90, 70, 255], dtype=np.uint8)

        mask = cv2.inRange(image, lower, upper)

        return mask

    @staticmethod
    def detectRedInHsv(image):
        lower = np.array([150, 45, 45], dtype=np.uint8)
        upper = np.array([210, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(image, lower, upper)

        return mask

    @staticmethod
    def getSignCircles(image):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=20, minRadius=25, maxRadius=50)

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

    def getSignsFromCircles(self, frame, circles):
        if circles is None:
            return None

        signs = []

        for (x, y, r) in circles:
            signs.append(frame[y - r - self.RECTANGLE_WIDTH:y + r + self.RECTANGLE_WIDTH, x - r - self.RECTANGLE_WIDTH: x + r + self.RECTANGLE_WIDTH])

        return signs

    def getSignsFromFrame(self, frame):
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsvMask = self.detectRedInHsv(hsvImage)
        # hsvMask = cv2.medianBlur(hsvMask, 3)

        if cv2.countNonZero(hsvMask) < self.COLOR_THRESHOLD:
            return None

        outputImage = cv2.medianBlur(hsvMask, 5)
        circles = self.getSignCircles(outputImage)

        return self.getSignsFromCircles(frame, circles)

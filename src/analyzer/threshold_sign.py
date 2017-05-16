import cv2
import numpy as np


class ThresholdSign:
    """ Represents thresholded sign"""

    def __init__(self, image):
        self.image = image
        self._result = self._create_from_image()

    def _create_from_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        cl1 = clahe.apply(gray)

        threshold = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

        return threshold

    def calculate_features(self):
        small_image = cv2.resize(self._result, (30, 30))
        small_image = small_image.reshape(1, 900)

        return np.float32(small_image)

    def get_result(self):
        return self._result

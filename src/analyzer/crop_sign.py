import cv2
from .threshold_sign import ThresholdSign
# from threshold_sign import ThresholdSign


class CropSign:
    """ Represents cropped sign image"""

    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape[:2]

        self._crop_sign()

    def _crop_sign(self):
        threshold_sign = ThresholdSign(self.image)

        im2, contours, hierarchy = cv2.findContours(threshold_sign.get_result(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if self._process_contour(contour):
                break

    def _process_contour(self, contour):
        if self._get_area_size(contour) < 20:
            return False

        [x, y, w, h] = cv2.boundingRect(contour)

        if self.width <= w or self.height <= h:
            return False

        if h < 20:
            return False

        self.image = self.image[y:y + h, x:x + w]

        return True

    def _get_area_size(self, contour):
        area = cv2.contourArea(contour)
        area_percent = 100 * area / (self.width * self.height)

        return area_percent

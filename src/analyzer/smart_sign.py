import cv2
from .blob import Blob


class SmartSign:

    def __init__(self, sign, knn_model):
        self.knn_model = knn_model
        self.sign = sign
        self.width = sign.width
        self.height = sign.height
        self.value = None

    def classify(self):
        blobs = self._get_blobs()
        blobs = sorted(blobs, key=lambda blob: blob.x)

        value = ""

        for blob in blobs:
            value += str(blob.get_value())

        try:
            self.value = int(value)
        except ValueError:
            return None

        return self.value

    def _get_blobs(self):
        digit_contours, thresh = self._get_contours(self.sign.image)
        blobs = []
        previous = None

        for contour in digit_contours:

            if self._is_blob(contour, previous) is False:
                continue

            [x, y, w, h] = cv2.boundingRect(contour)
            blob_image = thresh[y:y + h, x:x + w]

            blobs.append(Blob(blob_image, [x, y, w, h], self.knn_model))

        return blobs

    def _is_blob(self, contour, previous):
        if cv2.contourArea(contour) < 50:
            return False

        [x, y, w, h] = cv2.boundingRect(contour)

        if self.width == w or self.height == h:
            return False

        if 100 * w / self.width > 45:
            return False

        if 100 * h / self.height < 40:
            return False

        if h < 10:
            return False

        if previous is not None:
            [px, py, pw, ph] = previous

            if x > px and (x + w) < (px + pw):
                return False

        return True

    def _get_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours, thresh

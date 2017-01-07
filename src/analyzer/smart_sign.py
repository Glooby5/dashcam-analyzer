import cv2
import numpy as np


class SmartSign:

    def __init__(self, sign):
        self.sign = sign
        self.width = sign.width
        self.height = sign.height
        self._initialize_knn_model()
        self.value = None

    def classify(self):
        digit_contours, thresh = self._get_contours(self.sign.image)
        previous = None
        value = ""

        for contour in digit_contours:

            if self._is_blob(contour, previous) is False:
                continue

            [x, y, w, h] = cv2.boundingRect(contour)

            blob = thresh[y:y + h, x:x + w]
            value = value + self._get_digit_value(blob)

        self.value = int(value)

        return value

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

    def _initialize_knn_model(self):
        samples = np.loadtxt('general-samples.data', np.float32)
        responses = np.loadtxt('general-responses.data', np.float32)
        responses = responses.reshape((responses.size, 1))

        self.knn_model = cv2.ml.KNearest_create()
        self.knn_model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def _get_digit_value(self, image):
        roi = image
        roi_small = cv2.resize(roi, (10, 10))
        roi_small = roi_small.reshape((1, 100))
        roi_small = np.float32(roi_small)

        retval, results, neigh_resp, dists = self.knn_model.findNearest(roi_small, k=5)

        return str(int((results[0][0])))

    def _get_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours, thresh

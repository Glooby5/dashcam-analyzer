import cv2
import numpy as np


class Sign:

    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape[:2]

    def _crop_sign(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 1)
        im2, contours, hierarchy = cv2.findContours(threshold_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if self._get_area_size(contour) < 20:
                continue

            [x, y, w, h] = cv2.boundingRect(contour)

            if self.width <= w or self.height <= h:
                continue

            if h < 20:
                continue

            mask = self._get_circle_mask(contour)

            result = np.bitwise_and(self.image, mask)
            result = self._concat(result, mask)

            self.image = result[y:y + h, x:x + w]
            break

    def _get_area_size(self, contour):
        area = cv2.contourArea(contour)
        area_percent = 100 * area / (self.width * self.height)

        return area_percent

    def _get_circle_mask(self, contour):
        (x, y), radius = cv2.minEnclosingCircle(contour)

        center = (int(x), int(y))

        radius = int(radius)
        radius = (int)(radius * 0.85)

        mask = np.zeros_like(self.image)
        cv2.circle(mask, center, radius, (255, 255, 255), cv2.FILLED)

        return mask

    def _concat(self, img2, mask):
        blank = np.zeros_like(self.image)
        blank = cv2.bitwise_not(blank)

        rows, cols, channels = img2.shape
        roi = blank[0:rows, 0:cols]

        mask_inv = cv2.bitwise_not(mask)

        img1_bg = cv2.bitwise_and(roi, mask_inv)
        img2_fg = cv2.bitwise_and(img2, mask)

        dst = cv2.add(img1_bg, img2_fg)
        blank[0:rows, 0:cols] = dst
        return blank

import cv2
import numpy as np


class Blob:
    """ Represents numeric symbol"""

    def __init__(self, image, position, knn_model):
        [self.x, self.y, self.w, self.h] = position
        self.knn_model = knn_model
        self.image = image

    def get_value(self):
        roi = self.image
        roi_small = cv2.resize(roi, (10, 10))
        roi_small = roi_small.reshape((1, 100))
        roi_small = np.float32(roi_small)

        retval, results, neigh_resp, dists = self.knn_model.findNearest(roi_small, k=1)

        return int((results[0][0]))

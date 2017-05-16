import cv2
from .threshold_sign import ThresholdSign
from .crop_sign import CropSign


class TypeSign:
    """ Represent sign with type information"""

    def __init__(self, sign, knn_model):
        self.knn_model = knn_model
        self.sign = sign
        self.width = sign.width
        self.height = sign.height
        self.value = None
        self.type = None

        self.mapping = {
            1: "Zakaz chodcu",
            2: "Zakaz cyklistu",
            3: "Zakaz nakladaku",
            4: "Omezeni hmotnosti",
            5: "Zakaz predjizdeni",
            6: "Rychlost",
            7: "Zakaz stani",
            8: "Zakaz vlevo",
            9: "Zakaz vpravo",
            10: "Zakaz zastaveni",
            11: "Zakaz vjezdu)",
            12: "Omezeni rozmeru",
            13: "Omezeni hmotnosti"
        }

        self.classify()

    def classify(self):

        cropped_sign = CropSign(self.sign.original)

        sign = ThresholdSign(cropped_sign.image)
        roi_small = sign.calculate_features()

        retval, results, neigh_resp, dists = self.knn_model.findNearest(roi_small, k=1)

        self.type = retval

    def get_position(self):
        return self.sign.position

    def set_position(self, position):
        self.sign.position = position

    def get_original(self):
        return self.sign.original

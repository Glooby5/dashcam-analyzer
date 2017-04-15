import cv2
from .threshold_sign import ThresholdSign


class TypeSign:

    def __init__(self, sign, knn_model):
        self.knn_model = knn_model
        self.sign = sign
        self.width = sign.width
        self.height = sign.height
        self.value = None
        self.type = None

        self.mapping = {
            1: "chodci",
            2: "kolo",
            3: "nakladak",
            4: "nakladak_rozmer",
            5: "predjizdeni",
            6: "rychlost",
            7: "stani",
            8: "vlevo",
            9: "vpravo",
            10: "zastaveni"
        }

        self.classify()

    def classify(self):
        cv2.imshow("orig",self.sign.original )
        # cv2.waitKey(0)


        sign = ThresholdSign(self.sign.original)
        roi_small = sign.calculate_features()

        retval, results, neigh_resp, dists = self.knn_model.findNearest(roi_small, k=3)

        self.type = retval

    def get_position(self):
        return self.sign.position

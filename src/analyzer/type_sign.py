import cv2
from .threshold_sign import ThresholdSign
from .crop_sign import CropSign


class TypeSign:

    def __init__(self, sign, knn_model):
        self.knn_model = knn_model
        self.sign = sign
        self.width = sign.width
        self.height = sign.height
        self.value = None
        self.type = None

        # self.mapping = {
        #     1: "Zákaz vstupu chodců",
        #     2: "Zákaz vjezdu jízdních kol",
        #     3: "Zákaz vjezdu nákladních automobilů",
        #     4: "Z. v. voz. nebo souprav, jejichž délka přesahuje vyznačenou mez",
        #     5: "Zákaz předjíždění",
        #     6: "Nejvyšší povolená rychlost",
        #     7: "Zákaz stání",
        #     8: "Zákaz odbočování vlevo",
        #     9: "Zákaz odbočování vpravo",
        #     10: "Zákaz zastavení",
        #     11: "Zákaz vjezdu všech vozidel(v obou směrech)",
        #     12: "Z. v. voz., jejichž šířka přesahuje vyznačenou mez",
        #     13: "Zákaz vjezdu. voz., jejichž hmot. přesahuje vyznačenou mez"
        # }

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
        cv2.imshow("orig", self.sign.original)
        # cv2.waitKey(0)

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
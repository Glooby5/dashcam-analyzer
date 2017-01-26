from .sign import Sign
from .smart_sign import SmartSign


class Frame:
    def __init__(self, image):
        self.image = image
        self.signs = []

    def add_sign(self, position, knn_model):
        [x, y, w, h] = position
        sign = Sign(self.image[y:y + h, x:x + w], position)
        sign.crop_sign()

        classify_sign = SmartSign(sign, knn_model)
        classify_sign.classify()

        self.signs.append(classify_sign)

        return classify_sign

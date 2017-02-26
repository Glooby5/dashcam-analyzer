from .sign import Sign
from .smart_sign import SmartSign


class Frame:
    def __init__(self, image, number):
        self.image = image
        self.number = number
        self.signs = []
        self.fake = False
        self.time = None

    def add_sign(self, position, knn_model):
        [x, y, w, h] = position
        sign = Sign(self.image[y:y + h, x:x + w], position)
        sign.crop_sign()

        classify_sign = SmartSign(sign, knn_model)
        classify_sign.classify()

        self.signs.append(classify_sign)

        return classify_sign

    def get_sign_contain_point(self, point_x, point_y):
        for sign in self.signs:
            if sign.value is None:
                continue

            [x, y, w, h] = sign.get_position()

            if x - 100 <= point_x <= x + w + 100 and y - 100 <= point_y <= y + h + 100:
                return sign

        return None

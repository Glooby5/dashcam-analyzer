from src.analyzer.sign import Sign


class Frame:
    def __init__(self, image):
        self.image = image
        self.signs = []

    def add_sign(self, position):
        [x, y, w, h] = position
        sign = Sign(self.image[y:y + h, x:x + w])

        self.signs.append(sign)

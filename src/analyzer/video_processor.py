import cv2
import numpy as np
from .frame import Frame
from collections import deque
import copy

MAX_WIDTH = 150


class VideoProcessor:
    def __init__(self, cascade_file_name, video_file_name):
        self.video = cv2.VideoCapture(video_file_name)
        self.cascade = cv2.CascadeClassifier(cascade_file_name)
        self.frame_number = 0
        self.frame_counter = 0
        self._initialize_knn_model()
        self.actual_frame = None
        self.frame_deque = deque(maxlen=5)

    def get_next(self):
        ret, image = self.video.read()

        if ret is False:
            return False

        frame = Frame(image, self.frame_counter)
        self.actual_frame = frame

        sign_hits = self.cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=2, minSize=(20, 20))

        for (x, y, w, h) in sign_hits:
            if w > MAX_WIDTH or h > MAX_WIDTH:
                continue

            if w < 0 or h < 0:
                continue

            sign = frame.add_sign([x, y, w, h], self.knn_model)
            self.check_sign_value(frame, sign, [x, y, w, h])

        self.check_frame_signs(frame)
        self.frame_deque.appendleft(frame)
        self.frame_counter += 1

        return frame

    def check_frame_signs(self, frame):

        if len(frame.signs) > 0 or len(self.frame_deque) <= 1:
            return

        previous_frame = self.frame_deque[0]

        if self.frame_deque[0].fake or len(previous_frame.signs):
            return

        for sign in previous_frame.signs:
            fake_sign = copy.deepcopy(sign)
            [x, y, w, h] = fake_sign.get_position()

            previous_sign = self.frame_deque[1].get_sign_contain_point(x + w / 2, y + h / 2)

            if previous_sign is None:
                continue

            self.predict_next_sign(frame, fake_sign, previous_sign, [x, y, w, h])

    def predict_next_sign(self, frame, fake_sign, previous_sign, position):
        [x, y, w, h] = position

        [x1, y1, w1, h1] = previous_sign.get_position()
        [x2, y2, w2, h2] = [x + (x - x1), y + (y - y1), w + (w - w1), h + (h - h1)]

        fake_sign.set_position([x2, y2, w2, h2])

        frame.fake = True
        frame.signs.append(fake_sign)

    def check_sign_value(self, frame, sign, position):
        if sign.value is not None:
            return

        [x, y, w, h] = position

        for old_frame in self.frame_deque:
            if frame.number - old_frame.number > 5:
                break

            last_sign = old_frame.get_sign_contain_point(x + w / 2, y + h / 2)

            if last_sign is None:
                continue

            sign.set_value(last_sign.value)

            return

    def _initialize_knn_model(self):
        samples = np.loadtxt('general-samples.data', np.float32)
        responses = np.loadtxt('general-responses.data', np.float32)
        responses = responses.reshape((responses.size, 1))

        self.knn_model = cv2.ml.KNearest_create()
        self.knn_model.train(samples, cv2.ml.ROW_SAMPLE, responses)

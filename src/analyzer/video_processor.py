import cv2
import numpy as np
from .frame import Frame

MAX_WIDTH = 150


class VideoProcessor:
    def __init__(self, cascade_file_name, video_file_name):
        self.video = cv2.VideoCapture(video_file_name)
        self.cascade = cv2.CascadeClassifier(cascade_file_name)
        self.frame_number = 0
        self._initialize_knn_model()
        self.actual_frame = None

    def get_next(self):
        ret, image = self.video.read()

        if ret is False:
            return False

        if self.frame_number % 2:
            self.frame_number += 1
            return None

        frame = Frame(image)
        self.actual_frame = frame

        sign_hits = self.cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=2, minSize=(20, 20))

        for (x, y, w, h) in sign_hits:
            if w > MAX_WIDTH or h > MAX_WIDTH:
                continue

            if w < 0 or h < 0:
                continue

            frame.add_sign([x, y, w, h], self.knn_model)

        return frame

    def _initialize_knn_model(self):
        samples = np.loadtxt('general-samples.data', np.float32)
        responses = np.loadtxt('general-responses.data', np.float32)
        responses = responses.reshape((responses.size, 1))

        self.knn_model = cv2.ml.KNearest_create()
        self.knn_model.train(samples, cv2.ml.ROW_SAMPLE, responses)
